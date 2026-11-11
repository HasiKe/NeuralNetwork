// Vulkan compute backend: runs on any Vulkan 1.0 GPU (desktop, mobile, Linux,
// Windows, Android; macOS via MoltenVK).
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "backend.hpp"

namespace nn::gpu {

namespace {

const std::uint32_t kSpirv[] = {
#include "sgemm_spv.inc"
};

struct Push {
    std::int32_t M, N, K, lda, ldb, ldc, ta, tb;
    float alpha, beta;
};

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw Error(std::string("Vulkan ") + what + " failed (VkResult " + std::to_string(r) + ")");
}

class VulkanDevice final : public Device {
public:
    VulkanDevice(VkInstance inst, VkPhysicalDevice phys, std::uint32_t family)
        : m_inst(inst), m_phys(phys), m_family(family) {
        try {
            init();
        } catch (...) {
            destroy();
            throw;
        }
    }
    ~VulkanDevice() override { destroy(); }

    std::string name() const override { return m_name; }

    void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc) override {
        const std::size_t na = ta ? span(K, M, lda) : span(M, K, lda);
        const std::size_t nb = tb ? span(N, K, ldb) : span(K, N, ldb);
        const std::size_t nc = span(M, N, ldc);
        bool rebind = ensure(0, na) | ensure(1, nb) | ensure(2, nc);
        if (rebind) update_descriptors();
        std::memcpy(m_buf[0].map, A, na * sizeof(real));
        std::memcpy(m_buf[1].map, B, nb * sizeof(real));
        if (beta != 0 || ldc != N)  // keep padding between rows intact
            std::memcpy(m_buf[2].map, C, nc * sizeof(real));
        flush();

        Push pc{static_cast<std::int32_t>(M), static_cast<std::int32_t>(N), static_cast<std::int32_t>(K),
                static_cast<std::int32_t>(lda), static_cast<std::int32_t>(ldb), static_cast<std::int32_t>(ldc),
                ta ? 1 : 0, tb ? 1 : 0, alpha, beta};
        check(vkResetCommandBuffer(m_cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(m_cmd, &bi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout, 0, 1, &m_set, 0, nullptr);
        vkCmdPushConstants(m_cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(m_cmd, static_cast<std::uint32_t>((N + 63) / 64), static_cast<std::uint32_t>((M + 63) / 64), 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(m_cmd), "vkEndCommandBuffer");

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &m_cmd;
        check(vkResetFences(m_dev, 1, &m_fence), "vkResetFences");
        check(vkQueueSubmit(m_queue, 1, &si, m_fence), "vkQueueSubmit");
        check(vkWaitForFences(m_dev, 1, &m_fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        invalidate(2);
        std::memcpy(C, m_buf[2].map, nc * sizeof(real));
    }

private:
    struct Buf {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkDeviceSize cap = 0;
        void* map = nullptr;
        bool coherent = true;
    };

    void init() {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_phys, &props);
        m_name = std::string("Vulkan: ") + props.deviceName;
        m_atom = props.limits.nonCoherentAtomSize ? props.limits.nonCoherentAtomSize : 1;
        if (props.limits.maxComputeWorkGroupInvocations < 256 || props.limits.maxComputeSharedMemorySize < 8448)
            throw Error("device limits too small for the GEMM shader");

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = m_family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        check(vkCreateDevice(m_phys, &di, nullptr, &m_dev), "vkCreateDevice");
        vkGetDeviceQueue(m_dev, m_family, 0, &m_queue);

        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof kSpirv;
        si.pCode = kSpirv;
        check(vkCreateShaderModule(m_dev, &si, nullptr, &m_shader), "vkCreateShaderModule");

        VkDescriptorSetLayoutBinding binds[3] = {};
        for (std::uint32_t i = 0; i < 3; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.bindingCount = 3;
        dl.pBindings = binds;
        check(vkCreateDescriptorSetLayout(m_dev, &dl, nullptr, &m_dsl), "vkCreateDescriptorSetLayout");

        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &m_dsl;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pr;
        check(vkCreatePipelineLayout(m_dev, &pl, nullptr, &m_layout), "vkCreatePipelineLayout");

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m_shader;
        ci.stage.pName = "main";
        ci.layout = m_layout;
        check(vkCreateComputePipelines(m_dev, VK_NULL_HANDLE, 1, &ci, nullptr, &m_pipeline),
              "vkCreateComputePipelines");

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(m_dev, &dp, nullptr, &m_pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = m_pool;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &m_dsl;
        check(vkAllocateDescriptorSets(m_dev, &da, &m_set), "vkAllocateDescriptorSets");

        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = m_family;
        check(vkCreateCommandPool(m_dev, &cp, nullptr, &m_cpool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool = m_cpool;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(m_dev, &ca, &m_cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(m_dev, &fi, nullptr, &m_fence), "vkCreateFence");
    }

    void destroy() {
        if (m_dev) {
            vkDeviceWaitIdle(m_dev);
            for (auto& b : m_buf) release(b);
            if (m_fence) vkDestroyFence(m_dev, m_fence, nullptr);
            if (m_cpool) vkDestroyCommandPool(m_dev, m_cpool, nullptr);
            if (m_pool) vkDestroyDescriptorPool(m_dev, m_pool, nullptr);
            if (m_pipeline) vkDestroyPipeline(m_dev, m_pipeline, nullptr);
            if (m_layout) vkDestroyPipelineLayout(m_dev, m_layout, nullptr);
            if (m_dsl) vkDestroyDescriptorSetLayout(m_dev, m_dsl, nullptr);
            if (m_shader) vkDestroyShaderModule(m_dev, m_shader, nullptr);
            vkDestroyDevice(m_dev, nullptr);
            m_dev = VK_NULL_HANDLE;
        }
        if (m_inst) {
            vkDestroyInstance(m_inst, nullptr);
            m_inst = VK_NULL_HANDLE;
        }
    }

    void release(Buf& b) {
        if (b.map) vkUnmapMemory(m_dev, b.mem);
        if (b.buf) vkDestroyBuffer(m_dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(m_dev, b.mem, nullptr);
        b = Buf{};
    }

    // Grow buffer `slot` to hold n floats; returns true if it was reallocated.
    bool ensure(int slot, std::size_t n) {
        Buf& b = m_buf[slot];
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n ? n : 1) * sizeof(real);
        if (bytes <= b.cap) return false;
        release(b);
        const VkDeviceSize cap = bytes + bytes / 4;  // headroom against frequent regrowth
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = cap;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(m_dev, &bi, nullptr, &b.buf), "vkCreateBuffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(m_dev, b.buf, &req);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(m_phys, &mp);
        // Prefer host-visible + coherent (+ device-local on UMA / ReBAR systems).
        int best = -1, best_score = -1;
        for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if (!(req.memoryTypeBits & (1u << i))) continue;
            const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            int score = 1;
            if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score += 4;
            if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += 2;
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += 1;
            if (score > best_score) { best_score = score; best = static_cast<int>(i); }
        }
        if (best < 0) throw Error("no host-visible Vulkan memory type");
        b.coherent = (mp.memoryTypes[best].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = static_cast<std::uint32_t>(best);
        check(vkAllocateMemory(m_dev, &ai, nullptr, &b.mem), "vkAllocateMemory");
        check(vkBindBufferMemory(m_dev, b.buf, b.mem, 0), "vkBindBufferMemory");
        check(vkMapMemory(m_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
        b.cap = cap;
        return true;
    }

    void update_descriptors() {
        VkDescriptorBufferInfo info[3];
        VkWriteDescriptorSet w[3];
        for (std::uint32_t i = 0; i < 3; ++i) {
            info[i] = {m_buf[i].buf, 0, VK_WHOLE_SIZE};
            w[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = m_set;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &info[i];
        }
        vkUpdateDescriptorSets(m_dev, 3, w, 0, nullptr);
    }

    void flush() {
        for (auto& b : m_buf) {
            if (b.coherent) continue;
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, b.mem, 0, VK_WHOLE_SIZE};
            check(vkFlushMappedMemoryRanges(m_dev, 1, &r), "vkFlushMappedMemoryRanges");
        }
    }

    void invalidate(int slot) {
        Buf& b = m_buf[slot];
        if (b.coherent) return;
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, b.mem, 0, VK_WHOLE_SIZE};
        check(vkInvalidateMappedMemoryRanges(m_dev, 1, &r), "vkInvalidateMappedMemoryRanges");
    }

    VkInstance m_inst;
    VkPhysicalDevice m_phys;
    std::uint32_t m_family;
    VkDeviceSize m_atom = 1;
    VkDevice m_dev = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkShaderModule m_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_dsl = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkCommandPool m_cpool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    Buf m_buf[3];
    std::string m_name;
};

}  // namespace

std::unique_ptr<Device> create_vulkan(int index) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "nn";
    app.pEngineName = "nn";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ii, nullptr, &inst) != VK_SUCCESS) return nullptr;

    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    // Real GPUs first (discrete, integrated, virtual), software rasterisers last.
    struct Cand { VkPhysicalDevice d; std::uint32_t family; int rank; };
    std::vector<Cand> cands;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        std::uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qf.data());
        for (std::uint32_t q = 0; q < nq; ++q) {
            if (!(qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 0
                     : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1
                     : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 2 : 3;
            cands.push_back({d, q, rank});
            break;
        }
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.rank < b.rank; });
    if (index < 0 || static_cast<std::size_t>(index) >= cands.size()) {
        vkDestroyInstance(inst, nullptr);
        return nullptr;
    }
    const Cand& c = cands[static_cast<std::size_t>(index)];
    return std::make_unique<VulkanDevice>(inst, c.d, c.family);  // takes ownership of inst
}

}  // namespace nn::gpu
