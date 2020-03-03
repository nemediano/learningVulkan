#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <vector>
#include <cstdlib>
#include <cstdint> // Necessary for UINT32_MAX
#include <cstring>
#include <optional> 
#include <set>
#include <fstream>

// Helper function to load shadercode binary files into memmory
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    // If you cannot open file the file
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    // SInce you started at the and use you position to get the filesize
    size_t fileSize = (size_t) file.tellg();
    // Allocate a buffer (in bytes) of the size of the file
    std::vector<char> buffer(fileSize);
    // get back to the beggining of the file and read it
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    // Close the file and 
    file.close();
    // return the buffer
    return buffer;
}


const int WIDTH = 800;
const int HEIGHT = 600;

// List of names containing the validation layers
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// If we compile debug the validation layers are enabled
//#ifdef NDEBUG
    const bool enableValidationLayers = false;
// #else
//     const bool enableValidationLayers = true;
// #endif

// Query if the validation layers are in the available layers list
bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    std::cout << std::endl << "available layers:" << std::endl;
    for (const auto& layerProperties : availableLayers) {
        std::cout << "\t" << layerProperties.layerName << std::endl;
    }

    for (const char* layerName : validationLayers) {
       bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }
    return false;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const int MAX_FRAMES_IN_FLIGHT = 2;

class HelloTriangleApplication {
public:
void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

private:
    GLFWwindow* window;  // Handle to the GLFW window
    VkInstance instance; // Vulkan instance's handle (equivalent to GL Context ?)
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // Handle to the GPU
    VkDevice device;     // Handle to the logical device
    VkQueue graphicsQueue; // Handle to the graphics queue
    VkQueue presentQueue;  // Handle to the presentation queue (that might be different from the graphics one)
    VkSurfaceKHR surface;  // Handle to surface object
    VkSwapchainKHR swapChain; // Handle to the swapchain
    std::vector<VkImage> swapChainImages; // List of handles to the images on the swapchain
    std::vector<VkImageView> swapChainImageViews; // To be able to use the images in the swapchain
    std::vector<VkFramebuffer> swapChainFramebuffers; // one per each image in the swapchain
    VkFormat swapChainImageFormat;  // Of the images in the swapchain
    VkExtent2D swapChainExtent;     // ibidem
    VkRenderPass renderPass;        // Handle to the render pass
    VkPipelineLayout pipelineLayout; // Currently, not used but mandatory to have
    VkPipeline graphicsPipeline;    // Handle to the VK pieline object
    VkCommandPool commandPool;  // To create command buffers from it
    std::vector<VkCommandBuffer> commandBuffers; //The actual command buffers
    // One set of semaphores per frame in flight
    std::vector<VkSemaphore> imageAvailableSemaphores; // To signal when we adquire the image and is ready for rendering 
    std::vector<VkSemaphore> renderFinishedSemaphores; // To signal when image finished rendering and it is ready for presentation
    std::vector<VkFence> inFlightFences; // Perform GPU-CPU syncronization
    std::vector<VkFence> imagesInFlight; // Refer to each image in swapchain
    // keep track of the current frame so we know which pair of semaphores to use
    size_t currentFrame = 0;
void createSurface() {
    // The implementation is veri stright, since GLFW actually abstracted the platform
    // dependent code for us. We did not need to handle vulkan objects creations
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void createLogicalDevice() {
    // Use member handle too the phisical device to get the queue index
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    // fill structs to create the queues in the logical device
    // You need to give to each queue a priority, even if is the only one
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }   
    
    // If we need some GPU specific feature (we will fill it later)
    VkPhysicalDeviceFeatures deviceFeatures = {};
    // Now create the actual LOGICAL device
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // The info form the required queues (there is more than one)
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
    // The info form the phisical device (GPU)
    createInfo.pEnabledFeatures = &deviceFeatures;
    // Enable extensions required in the logical device
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    // Prevoius versions of VK distinguis between instance and device specific validation layers
    if (enableValidationLayers) {
        // These attributes are ignored in new VK implementations
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }
    //All the info is filled, now create the Logical device
    // Again, use member variable physicalDevice
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
    // We create the logical device, but we still need a handle to his graphics queue
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    // and now a handle to the presentation queue
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    // Logic to find graphics queue family
    QueueFamilyIndices indices;
    // Query for the available query families
    uint32_t queueFamilyCount = 0;
    // How many?
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    // which ones
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    // Loop searchin for a graphics queue
    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        // Check that the family support graphics operations
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        // If supports rendering to a surface
        if (presentSupport) {
            indices.presentFamily = i;
        }
        // We found it return early
        if (indices.isComplete()) {
            break;
        }

        i++;
    }    

    return indices;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    std::cout << "Device extensions:" << std::endl;
    std::cout << "Required extensions:" << std::endl;
    for (auto extension : requiredExtensions) {
        std::cout << "\t" << extension << std::endl;
    }
    std::cout << "Available extensions:" << std::endl;
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
        std::cout << "\t" << extension.extensionName << std::endl;
    }

    return requiredExtensions.empty();
}

// Check if this GPU is suitable for our needs
bool isDeviceSuitable(VkPhysicalDevice device) {
    
    // VkPhysicalDeviceProperties deviceProperties;
    // VkPhysicalDeviceFeatures deviceFeatures;
    // vkGetPhysicalDeviceProperties(device, &deviceProperties);
    // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);


    // return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && // Dedicated GPu
    //        deviceFeatures.geometryShader; // that supports geometry shaders
    // If we have the needed queues
    QueueFamilyIndices indices = findQueueFamilies(device);
    // If the device supports rendering
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    // If the swapchain is adecuate
    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        // A swapchain is adequate if it has at least one image format and one presentation mode
        // compatiobel with the surface
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    // Loook for a prefered format
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    // That failed just use the first one
    return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {

    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = {WIDTH, HEIGHT};

        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

void createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    // Minimum number of images required for this swapchain to function plus one (se we do no wait on driver)
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    // We also need to make sure we do not exced the maximum
    // Funny a zero, in capabilities.maxImageCount means that there is no maximum
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
        
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    // We are finally ready to actually create the swapchain
    // 1. Fill structure with details
    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1; // unless developing a stereoscopic 3D application
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // Images are to present not for post process
    //Check if we can use the same queue to draw (graphics) and to present (render to screen)
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    // Different queues
    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else { // They are the same queue
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }
    // Mean no transformation. We are just presenting not flippping nor rotating
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    // Do not blend with other windows in the window system
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // The one you retived on top
    createInfo.presentMode = presentMode;
    // Unless you want to send the pixels obscured by other windows for some reason
    createInfo.clipped = VK_TRUE; // We dont care of non render pixels (obscured)
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    // 2.- Finally create the swapchain
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }
    // 3.- Get hanldes to the images in the swapchain
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
    // Store the image format and extent for later use
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void createImageViews() {
    // Store one view per imge in the swapchain
    swapChainImageViews.resize(swapChainImages.size());
    // For each of them create a view
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        // Populate struct create info
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        // You can swizzle the channels (we stick to the default)
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        // Our images will be used as render target, no mipmaps no multiple layers
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        // 2.- Try to create image view
        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

//Select a GPU to use
void pickPhysicalDevice() {
    // Query how many VK capable GPU we have
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        // We have none trow exception
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    // Get the list of the available GPUs
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    // See if we find a suitable GPU
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }
    // We did not find it throw exception
    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    // 1.- Query the surface capabilities of this device
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    // 2.- Query the supported formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }
    // 3.- Query for presentation modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }
    return details;
}


void initWindow() {
    glfwInit();
    // Do not create an OpenGL context (since we will use vulkan)
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Disable resize for now, since in VK requires extra care to handle
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    // Ask GLFW to create the window
    window = glfwCreateWindow(WIDTH, HEIGHT, "Hello Triangle", nullptr, nullptr);
}

void initVulkan() {
    // The instance is the connection between your application and 
    // the Vulkan library and creating it involves specifying some details about your application
    // to the driver.
    createInstance();
    // Create surface object, sbefore since it can affect the device selection
    createSurface();
    // See if there is a phisical device GPU that we can use
    pickPhysicalDevice();
    // Create a logical device with his corresponding queues
    createLogicalDevice();
    // Create swapchain that is compatible with the logical device
    createSwapChain();
    // We need to create a view for each image in the swapchain that we are going to use
    createImageViews();
    // Before pieline creation
    createRenderPass();
    // Create the pieline used in the program
    createGraphicsPipeline();
    // After the swapchain and after the pipeline
    createFramebuffers();
    // Now, that we have the pipeline, create the command buffers. Start with command pool
    createCommandPool();
    // Now create and record the command buffers
    createCommandBuffers();
    // needed to sync the rendering and the swapchain
    createSyncObjects();

}

void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = 0; // Optional
    /*
        Command buffers are executed by submitting them on one of the device queues,
        like the graphics and presentation queues we retrieved.
        Each command pool can only allocate command buffers that are submitted on a
        single type of queue.
        We're going to record commands for drawing, which is why we've chosen the
        graphics queue family.
    */
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void createCommandBuffers() {
    //One command buffer for each swapchain image
    commandBuffers.resize(swapChainFramebuffers.size());
    // 1.- Create info structure for the command buffers
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();
    // 2.- Create the command buffer
    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
    /******************************************/
    /* Record commands in the command buffer **/
    /******************************************/
    for (size_t i = 0; i < commandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional
        // Mark the beginning of the command buffer
        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        // Start the render pass
        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];
        // The whole framebuffer in the swapchain
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        // Clear color for this framebuffer at the start of the render pass
        VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;
        // Begin the render pass in this command buffer
        // Since the drawing commands are in the fb itself we choose to provide as inline
        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        // Othe command this binds the pipeline
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        // Tell him to actually draw a triangle!!
        /*        
            vertexCount: Even though we don't have a vertex buffer, we technically still have 3 vertices to draw.
            instanceCount: Used for instanced rendering, use 1 if you're not doing that.
            firstVertex: Used as an offset into the vertex buffer, defines the lowest value of gl_VertexIndex.
            firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
        */
        vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);
        // Mark the end of the render pass
        vkCmdEndRenderPass(commandBuffers[i]);
        // Now we finished issue the commands try to record this command buffer
        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

}

void createFramebuffers() {
    // Resize to hold as much as the images
    swapChainFramebuffers.resize(swapChainImageViews.size());
    // Iterate creating each framebuffer
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void createGraphicsPipeline() {
    /************************************/
    /* Programable part of the pipeline */
    /************************************/
    auto vertShaderCode = readFile("shaders/vert.spv");
    auto fragShaderCode = readFile("shaders/frag.spv");
    // Create the vk shader module objects from files
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
    // Shader creation
    // First vertex shader
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule; // <= link to the module object
    vertShaderStageInfo.pName = "main"; // Entry point of this shader
    // Then fragment
    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    // Create an array to hold the two shader info structures
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    /************************************/
    /* Fixed part of the pipeline       */
    /************************************/
    // Vertex's shader input configuration (No vertex data now)
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional
    // Input assamblly: Which kind of geometry we will generate and if we allow for reset
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    // Specify the view port (usually the whole surface)
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapChainExtent.width;
    viewport.height = (float) swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    // Clipping rectangle (the whole framebuffer)
    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;
    // Now create the viewport stage ()
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    // Create rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE; // Weird behaviour for shadow mapn do not use it
    rasterizer.rasterizerDiscardEnable = VK_FALSE; //Disable the frambuffer output (false to keep it enable)
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // Do we ouput lines, points or polygons
    rasterizer.lineWidth = 1.0f; //in units of fragments Default mode other requires enable extra features
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // Do we do face cull?
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // winding order
    rasterizer.depthBiasEnable = VK_FALSE; // Some bias in the depth (for example to avoid petepanning in shadow maps)
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
    // Now multisampling (disable for now)
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional
    // Color blending per buffer attachment (we only have one)
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    // Deafult alpha blenidng
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    // Color blend global to all the framebuffers
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional
    // Small part of states of the pipeline that can change wo recreation
    VkDynamicState dynamicStates[] = {
                                        VK_DYNAMIC_STATE_VIEWPORT,
                                        VK_DYNAMIC_STATE_LINE_WIDTH
                                     };
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
    // Pipeline layout creation (to pass uniform values)
    // Not used now, we will keep it empthy his creation info
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {}; // Creation info
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // Optional
    pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    // Finally create the pipeline here
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;  //Just two shader stages
    pipelineInfo.pStages = shaderStages;
    // All the other structs
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr; // Optional
    // Use the layout 
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass; // The one from the previous method now in member
    pipelineInfo.subpass = 0;
    // VK let you inherit form other pipelines since we just have one we set them to null/none
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    // Destroy vk shader module objects afetr pipeline creation thay are not needed anymore
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void createRenderPass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // Do we clear before loading or not
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; //After render do we keep the content or we do not care
    // Same but for stencil (since we dont use, we say we do not care)
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; 
    // Layout of the framebuffer
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // We do not care befroe (since we clear)
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // We do care later (something compatible with our swapchain)
    // If we do multiple render passes, we need to configure separatelly
    // We need to configure a render atechment that we are going to use in the subpass
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0; //only one render target
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // in which layout at the time of the pass
    // Now configure the render pass (or subpass)
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1; // On single frameburre atched
    subpass.pColorAttachments = &colorAttachmentRef; // The one (or multiples) that was configured like this
    // We need to create a subpass dependency
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    //Finally create the subpass (by using both struct form above)
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1; //Add the suppass dependency
    renderPassInfo.pDependencies = &dependency;
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
    

}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    // 1.- Create infor structure for VK shader mnodeul object
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size(); // Vector size is alrady in bystes char == byte
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); // std:vector are tighly packed and aligned
    // 2.- Now creaet the VK object
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    // And return handle to caller
    return shaderModule;
}

void createSyncObjects() {
    // Resize one set per frame in flight
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    // Fences
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

    // 1.- Fii info creation structures
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; //So they are created as finished

    // Since the structure is actually very simple and no specific info,
    // we use it to 2.- create both semaphores for each frame
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS
        ) {

            throw std::runtime_error("failed to create syncronization object for a frame");
        }
    }
    
}

void drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    
    /*
    The first two parameters of vkAcquireNextImageKHR are the logical device and the swap chain
    from which we wish to acquire an image. The third parameter specifies a timeout in nanoseconds
    for an image to become available. Using the maximum value of a 64 bit unsigned integer disables
    the timeout.

    The next two parameters specify synchronization objects that are to be signaled when the
    presentation engine is finished using the image. That's the point in time where we can start
    drawing to it. It is possible to specify a semaphore, fence or both. We're going to use our
    imageAvailableSemaphore for that purpose here.

    The last parameter specifies a variable to output the index of the swap chain image that has
    become available. The index refers to the VkImage in our swapChainImages array. We're going
    to use that index to pick the right command buffer.
    */
    uint32_t imageIndex;
    vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
    //After adquiring the image
    // Check if a previous frame is using this image (i.e. there is its fence to wait on)
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    // Mark the image as now being in use by this frame
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];


    // Submit the command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // To which semaphore wait
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    // which buffer to submitt
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    // Which semaphore signal one you finish
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    // Submmit the queue
    vkResetFences(device, 1, &inFlightFences[currentFrame]); //Just before using it
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    /****************/
    /* Presentation */
    /****************/
    // to which semaphore wait before presentation
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    // to which swapchain present images
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // Optional
    vkQueuePresentKHR(presentQueue, &presentInfo);
    vkQueueWaitIdle(presentQueue);
    //Advance to the next frame
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}


void mainLoop() {
    // Main loop, keep running until a close call is issued
    while (!glfwWindowShouldClose(window)) {
        // Pool any event that happened
        glfwPollEvents();
        //Render function
        drawFrame();
    }
    // Since drawing is async, we might be still rendering when we try to close and clean resources
    // wait until the device finished operation before cleaning
    vkDeviceWaitIdle(device);
}

void cleanup() {
    // When all the render is none and no more sync is needed
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    // No specific order, just after we are not drawing anymore
    vkDestroyCommandPool(device, commandPool, nullptr);
    // Before image view and render pases in which depends on
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    // The pipeline must be destroyed before the piepline layout
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    // Since, its global object used in several methods created by us
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    // Same here no particular order
    vkDestroyRenderPass(device, renderPass, nullptr);
    // Imageviews were created for us 
    // (as oppouse from the images, that were created by the swapchain object)
    // so we need to destroy them (Obviously, before the swapchain object)
    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    // Destyroy swapchain before device
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    // Logical devices do need to be destroyed
    vkDestroyDevice(device, nullptr);
    // Surface destruction (GLFW create it, but does not have function to destroy it). 
    // Therefore, we use standar VK destruction function
    // Also, surfce needs to be destroyed before the instance
    vkDestroySurfaceKHR(instance, surface, nullptr);
    // The instance should be destroyed right before application exits
    // we create this function fo that purpose specifically
    vkDestroyInstance(instance, nullptr);
    // Now, that VK instance is destroyed, we can alse destroy the window
    glfwDestroyWindow(window);
    glfwTerminate();
}

void createInstance() {
    // Before creating instance, see if we have validation layers
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }
    // Struct to communicate the type of app we are going to use to the dirver
    // its optioonal.
    VkApplicationInfo appInfo = {}; 
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Mandatory struct that tells the Vulkan driver which global extensions and
    // validation layers we want to use
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo; // <- We chain the previous struct here
    // We need the the list of needed extensions
    // VK is API agnostic so we need to quety the GLFW to tell us which one we need
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    // Now, that you now which extension do you need, put the info in the struct
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }
 
    // Create the VK instance, if something fails abort
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create instance!!");
    }
    std::cout << "Instance extensions:" << std::endl;
    // Just an extra exercise: 
    // 1.- List the required extensions
    std::cout << std::endl << "required extensions:" << std::endl;
    for (size_t i = 0; i < glfwExtensionCount; ++i) {
        const char* extension = glfwExtensions[i];
        std::cout << "\t" << extension << std::endl;
    }
    // 2.- Query for the number of available extensions:
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
    // Print all the available extensions
    std::cout << "available extensions:" << std::endl;
    for (const auto& extension : extensions) {
        std::cout << "\t" << extension.extensionName << std::endl;
    }
}

}; // HelloTriangleApplication class

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}