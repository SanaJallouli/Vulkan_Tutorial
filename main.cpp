#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>

#define GLM_FORCE_RADIANS
// force glm to use a version of vec and mat that has the alignment requirements already specified. otherwise need to preceid the non aligned member of uniform with alignas(16) .
// Note : this method can break down if you use nested structures , in that case specify the alignment yourself !
// it may be better to always specify the alignment yourself to avoid weird behaviours.
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include "glm/glm.hpp" // provides C++ types that matches the vector types in shaders.

#include <glm/gtc/matrix_transform.hpp> // exposes functions that ca be used to generate model transformations

#include <chrono> //precise timekeeping

#include <iostream>

#include <stdexcept>

#include <cstdlib> // provide the EXIT_SUCESS and EXIT_FAILURE macros

#include <iostream>

#include <vector>

#include <fstream>

#include <optional>

#include <set>

#include <random>

#include <array>

#include <cstdint> // Necessary for uint32_t

#include <limits> // Necessary for std::numeric_limits

#include <algorithm> // Necessary for std::clamp

#include <iostream>

#include <vulkan/vulkan_beta.h>

using namespace std;

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector <const char * > validationLayers = { "VK_LAYER_KHRONOS_validation"};

const std::vector <const char * > deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
  std::optional < uint32_t > graphicsFamily;
  std::optional < uint32_t > presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct Particle {
glm::vec2 position;glm::vec2 velocity; glm::vec4 color;


    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Particle);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle, color);

        return attributeDescriptions;
    }

};

int PARTICLE_COUNT = 8192;

// vertex buffer
// vertex buffer : associated with vertex buffer memory :
/*
 //
 - specify the vertex data in an array , data in cpu memory to be then transferred to the vertex buffer (in GPU)
 - create vertex buffer :
    - allocate memory to the vertex buffer memory using the right memory properties suitable to the application and the type of vertex
    - associate the vertex buffer handler to the memory.
    - copy the vertex data from the vertex data array into the vertex buffer (using staging buffer or mapping the memory)
 - Bind the vertex buffer to the command buffer during the render pass
 - Specify the format of the vertex data that will be passed to the vertex shader during the graphics pipeline creation : the binding description (describes at which rate to load the data : whether to move after each vertex or each entry) and attribute descriptions (ex in our example we have an array of two descriptions :  color and position, for which we specify layout location, format , offset .. )
 
 */

// Define the vertex data with the attribute to use (here position and color ) :
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  // once the vertex data is uploaded to GPU, Vulkan needs to know how to pass the data to the vertex shader : 2 structures needed to convey this info:

  // 1. first structure :VkVertexInputBindingDescription
  // describes at which rate to load data from memory throughout the vertices. It specifies the number of entries and whether to move to the next data entry after each vertex of after each instance.

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription {};
    // we have all the vertex data in one array , so we have one binding.
    bindingDescription.binding = 0; // index of the binding in the array of bindings.
    bindingDescription.stride = sizeof(Vertex); // number of bytes from one entry to the next
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // move to next data entry after each vertex or after each instance. (here per vertex )
    return bindingDescription;
  }

  // 2nd structure : VkVertexInputAttributeDescription (one attribute description structure for each attribute, for us it is 2 : color and position )
  // specifies how to handle vertex input : describe how to extract a vertex attribute from a chunk of vertex data originating from binding description.

  // we have 2 attributes : color and position, so we need two attribute description structs.
  static std::array < VkVertexInputAttributeDescription, 3 > getAttributeDescriptions() {
    std::array < VkVertexInputAttributeDescription, 3 > attributeDescriptions {};

    // position :
    attributeDescriptions[0].binding = 0; // from which binding the per vertex data comes.
    attributeDescriptions[0].location = 0; // the location directive of the input in the vertex shader : layout(location = 0) in vec2 inPosition;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; // type of data
    /* ex : the amount of color channels matches the number of components in the shader data types.If you use more channels, they will just be discarded.
     If you use less channels, than the BGA will default to (0, 0, 1).
     • float: VK_FORMAT_R32_SFLOAT
     • vec2: VK_FORMAT_R32G32_SFLOAT
     • vec3: VK_FORMAT_R32G32B32_SFLOAT
     • vec4: VK_FORMAT_R32G32B32A32_SFLOAT*/
    // it implicitly defines the byte size of attribute data.

    attributeDescriptions[0].offset = offsetof(Vertex, pos); // pos : position attribute
    // define the number of bytes since the start of the per-vertex data to read from. This is automatically calculated using the macro.

    // color :
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1; // layout(location = 1) in vec3 inColor;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    // texture coordonates: the top -left corner is 0,0
    // the bottom right corner is 1,1
    // to see addressing mode in aaction : use coordinates belowe 0 and above 1
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }

};

  // An array of vertex data (interleaving vertex attributes )
  const std::vector<Vertex> vertices = {
  {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
  {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
  {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
  {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
  };

// vertex indices :
const std::vector < uint16_t > indices = { // uint_16 or uint32 depending on the number of vertices to use.
  0,1,2,2,3,0};

// uniform buffer :
/*
Resource descriptors : way for shader to access resources like buffer and images. ex of resource descriptor is uniform buffer.

- 1. Specify the data (here in the form of struct) that will be then transferred to the uniform buffer (in GPU) and accessed by the shader

- 2. create descriptor set layout object containing the type and details of the descriptor sets (one object hold the binding of all the descriptor sets) :
specify details about every descriptor (type which is uniform buffer in our case, how many descriptors which is 1 here , where it will be used which is in vertex shader here), the binding that will be used in the shader, and use these details to create the descriptor set layout object.
now we have descriptorSetLayout object that holds the info about all the descriptors binding (how every descriptor is to be accessed, where ... ). we can later have many descriptor sets (the actual descriptor set object associated with buffer and descriptor pool ) using the same descriptor layout.

-3. create uniform buffer : each uniform buffer has memory associated with it and mapped memory  asoosicated with it :
       - map the buffer memory and keep a pointer to where the data is written : memeory stays mapped
       - associate the uniform buffer memory to the memory associate with it
we want to have as many uniform buffer as we have frame on flight , so we have a vector of uniform buffers, a vector of memory associate with it, a vector of mapped mempry associated with it.

-4. create Descriptor pool to allocate descriptor sets :
       specify the type of descriptor sets the pool is going to generate (here uniform buffers), the number of descriptor sets and use these info to create the descriptor pool. this deos not specify the layout. may have same pool, generating descriptor sets with different layouts. this is specified in the creation of the description set.

-5. use descriptor pool to create descriptor sets objects :
we will create as many descriptor set objects as we have frame on flight, all using the same layout
we define a vector of descriptor sets objects.
       - allocate descriptor sets : specify the pool from which to allocate them (contain the info that the type of descriptor is uniform buffer), the number of descriptor set to allocate, the descriptor layout.
           now we have a vector of allocated descriptor sets from the descriptor pool.
       - for each descriptor set in the vector of descriptor sets :
           - how the configuration of the descriptor is updated : how to write the descriptor set:
               -specify the destination binding (our binding for this layout is 0, specify again the type of descriptor, how many descriptor sets to update)
               - specify the buffer associated with it
               - specify the region within the buffer that contains the data of the descriptor set : offset + range (sze of the uniform data )
           - use the update infos to call update

-6. in the record command buffer, bind the right descriptor set for each frame
-7. if you want to update the data that is in the uniform buffer:
   - make the update of the data in the draw frame so that it is done every frame: mskr the update and memcopy the new array into the mapped memroy
*/

// the data we want the vertex shader to have :
// Need to provide details about every descriptor binding used in the shaders (like where we provided details about every vertex attribute and its location index)
// this dta must match the uniform defintion in the shader :
// - 1. must use the same tyope in both
// - 2 alignment requirements : vulkan expects the data in the memory to be aligned in a certain way:
// ex: mat4 has to be aligned by 4N (16 bytes) : all the offsets have to be mutliple of 16 for the mat4 : here it is alread y the case, model has offset of 0, view has offset of 64 and proj has offset of 128 , sp all are multiple of 16.
// imagine if we added vec2 before these mat4 , vec 2 is only 8 bytres, vec2 has offset of 0, model has offset of 8, view has offset of 72 and view has offset of 136, they are not multiple of 16.
// there is 2 ways to fix this : add alignas (16) before each of the mat4 , or use glm directive.
// the problem with glm directive is that it does n t work with nested structures. so better be safer and always use the alignas

// pass the details of the swap chain
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

// combined image sampler:
// another type of descriptors like uniform buffers , we have combined image sampler.
/*
 combined image sampler makes it possible for shaders to access image ressoures through sampler object
 
 we need to modify the descriptor layout, descriptor pool and descriptor set to include this type of descriptor
 */


class HelloTriangleApplication {
  public: void run() {

    initWindow(); // initialize GLFW and create a window

    initVulkan(); // Initialize the vulkan library : create an instance (the connexion between the app and vulkan library ). = specify some details about your app to the driver

    mainLoop();

    cleanup();
  }

  uint32_t currentFrame = 0;

  bool framebufferResized = false;

  void initWindow() { // will store the Vulkan objects and function to initiate each of them, called from
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow * window, int width, int height) {
    auto app = reinterpret_cast < HelloTriangleApplication * > (glfwGetWindowUserPointer(window));
    app -> framebufferResized = true;
  }

void initVulkan() {

      // a. create instance
      createInstance();

      // b. create surface
      createSurface();

      //c. select graphics card in the system that supports the features we need (can be any number of graphic cards used simultaneously)
      pickPhysicalDevice();

      //d. Create logical device to interface with the logical device
      createLogicalDevice();

      // e.
      createSwapChain();

      // f.to use any VkImage, including the ones in the swap chain, in the render pipeline , we have to use ImageViews
      // it describes how to access the image, and which part to access.
      createImageViews();

      //g. need to tell vulkan about the framebuffer attachment that will be used while rendering, specify how many color buffer and depth buffer there is, how many samples to use for each of them and how their content should be handled throught the rendering operations. all these are wrapped into a render pass object
      createRenderPass();

      // h.Create descriptor set layout :
      // descriptor : a way for shaders to access resources like buffers and images
      // Descriptor layout : specifies the type of resources that are going to be accessed by the pipeline
      // Descriptor set : specify the actual buffer or image ressources that will be bound to the descriptor
      // descriptor set is then bound for the drawing command.
      // NOTE : THIS IS NOT THE MOST EFFICIENT WAY TO PASS SMALL BUFFERS TO SHADERS (the most efficient is PUSH CONSTANT )
      // this specifies the type of descriptors that can be bound
      // later we need to create a descriptor set for each vkBuffer ressource to bind it to the uniform buffer descriptoon
      createDescriptorSetLayout();

      // i.
      createGraphicsPipeline();

      // j.
      createComputePipeline();

      //k . Create frame buffers : we have set a render pass to expect a single framebuffer with same format as the swap chain image, now we need to create a framebuffer.
      createFramebuffers();
      // commands (ex: drawing operations and memory transfer) are not executed directly using function calls. all the operations to be performed are recorded in command buffer object.
      // advantage 1 : when we are ready to tell vulkan what we want to do, all the commands will be submitted together and Vulkan can more efficiently process the commands.
      // advantage 2 : command recording can happen in different threads
      // to be able to create command buffer, we need first to create command pool, they are responsible of managing the memory for buffers , so that command buffer can be allocated from them .

      //l. create command pool
      createCommandPool();

      // m.
      createTextureImage();

      //n.
      createTextureImageView();

      // o.
      createTextureSampler();

      // p. buffer : region in memory to store arbitrary data that the graphics card can read. They do not automatically allocate memoery for themselves.
      createVertexBuffer();

      // q.
      createIndexBuffer();

      //r.
      createShaderStorageBuffers();

      //s. create uniform buffer :
      // the buffer that contains the UBO data for the shader.
      // New data is copied to this buffer every frame, this buffer doesn'tneed a staging buffer . It would degrade performance instead of ameliorating it.
      createUniformBuffers();

      //t. Like command buffers, descriptor sets cannot be created direclty, they must be allocated from a pool = descriptor pool.
      createDescriptorPool();

      // u. Allocate the descriptor sets themselves : cannot be done directly , need to use descriptor pool
      createDescriptorSets();

      // v.
      createCommandBuffers();

      //w.
      createComputeCommandBuffers();

      //x.
      createSyncObjects();
}
    
    // images are accessed via image views not direclty. so we need to create image view for the texture image.
    VkImageView textureImageView; // hold the image view associated with the image where we put the pixels of the texture we loaded.
    void mainLoop() {
      while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrameWithComputePass();
      }

      vkDeviceWaitIdle(device);
    }


  void cleanup() {
   // cleanup SwapChain:
      for (auto framebuffer: swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
      }

      for (auto imageView: swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
      }

      vkDestroySwapchainKHR(device, swapChain, nullptr);

    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroyBuffer(device, uniformBuffers[i], nullptr);
      vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
  }

    
    // Helper functions:

    // create shader module : take code of the shader and create a shader module from it
    // pass the code as buffer of bytecode
    VkShaderModule createShaderModule(const std::vector<char>& code) {

        // 1. spcify a pointer to the buffer with the bytecode and its lenfh
        // spicify these ifo into the appropriate structure : VkShaderModuleCreateInfo
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        // the size of the bytecode is specified in bytes but the bytecode pointer is a uint_32
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        //2. create the vkShaderModule
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr,
            &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    // Helper function to find the right settings for the best possible swap chain
        // there are three types to determine

        // 1. surface format = color depth :
        //  VkSurfaceFormatKHR entries has format and color space members
        // format specify the color channel and type
        // the color space specify if the SRGB color space is supported or not
        //  we want to use SRGB if it is available (more accruate perceived color)
        // need to use srgb color format ex : VK_FORMAT_B8G8R8A8_SRGB
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const
            std::vector<VkSurfaceFormatKHR>& availableFormats) {
            // iterate through the list of formats available and check if our preferred combination is available

            for (const auto& availableFormat : availableFormats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
                    availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    return availableFormat;
            }
            // if that ideal combination is not available
            // it is ok to just go with the first one
            // or rank them
           return  availableFormats[0];
        }


        //2. Presentation mode : the most important setting for the swap chain
        // represents the actual conditions for showing images in the screen

        // we want to use (triple buffering mode) VK_PRESENT_MODE_MAILBOX_KHR, otherwise use the one that is guarantee available that is fifo
        VkPresentModeKHR chooseSwapPresentMode(const
            std::vector<VkPresentModeKHR>& availablePresentModes) {
            for (const auto& availablePresentMode : availablePresentModes) {
                // TRIPLE BUFFERING mode : instead of blocking the queue, if the queue is full, and a new image is to be inserted, replace older images from the queue : render frames as fast as possible
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                }
            }
            return VK_PRESENT_MODE_FIFO_KHR; // the only one that is guarantee to be supported
            // fifo : images are queued , monitor renders images from front and put the rendered images in the back of the queue
                // program waits if queue is full
        }


        //3. swap extent : resolution of the swap chain images
        // almost always equal to the resolution of the window we are draweing to in pixels/

        // vulkan tells us to maych the resolution of the window by setting the width and hight in the currentExtent
        // capatbilities has member current extent that uses pixel
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& // the range of possible resolutions
            capabilities) {

            // some window manager allow us to set up a special value = the maximum value of uint32_t
            // that n the minImageExtent and maxImageExtent bounds .
            // we specify the resolution in pixel

            // if the capabilities does not support max 32 int, this means it will not let us pick the reolution that best  match the best resolution
        // means it matches the width and hight of the current extend
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            {
                return capabilities.currentExtent;
            }
            else // it is max 32 , we pick the resolution that best matches the window
            {
                int width, height;
                glfwGetFramebufferSize(window, &width, &height); // query the resolution of the window in pixel


                VkExtent2D actualExtent = {
                    static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height)
                };

                // use clamp to bound the values of width and height between the alloawed min and max supported
                actualExtent.width = std::clamp(actualExtent.width,
                    capabilities.minImageExtent.width,
                    capabilities.maxImageExtent.width);
                actualExtent.height = std::clamp(actualExtent.height,
                    capabilities.minImageExtent.height,
                    capabilities.maxImageExtent.height);

                return actualExtent;

            }
        }


  // pass the details of the swap chain
  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector < VkSurfaceFormatKHR > formats;
    std::vector < VkPresentModeKHR > presentModes;
  };

  // function to populate the struct of the swap chain details
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    //all query function have these two parameter: Physical device / VkSurfaceKHR window surface
    // they are the core components of the swap chain and thus are taken into account when determining the suppoprt capability

    // surface capabilities query
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &
      details.capabilities);

    // surface format query
    // list struct : 2 step to fill the count then the data
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, & formatCount,
      nullptr);

    if (formatCount != 0) { // make sure the vector is resized to hold all the available formats.
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &
        formatCount, details.formats.data());

    }

    // query supported presentation mode
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &
      presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &
        presentModeCount, details.presentModes.data());

    }

    return details;
  }

    // helper function to select the device
  bool isDeviceSuitable(VkPhysicalDevice device) {
    // query for some details: ex: name, type and supported vulkan version
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, & deviceProperties);

    // query for optional features : ex : texture compression, 64 floats , multi viewport rendering (for vr)
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, & deviceFeatures);

    // return the first one that checks the condition
    // alternatively we can give each device a score and choose the one with highest score.

    /* return deviceProperties.deviceType ==
         VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
         deviceFeatures.geometryShader;
         */

    // use findQueueFamilies to see if the device has the queue families we need (here graphics queue family is implemented )
    QueueFamilyIndices q = findQueueFamilies(device);

    //1. make sure device has graphics possibilities : has graphics queue
    //2. make sure device has presentation capabilities : has presentation queue

    //3. make sure device support swapchain , swapchain is an extension

    // a. list all the extensions and see of the required ones are there

    // 1.first we need to know how many extensions are they to create an array to hold them :
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &
      extensionCount, nullptr);

    // 2.now we have the number of parameters, we can allocate an array to hold the extension details :
    // each VkExtensonProperties struct contains the name and version of an extension.
    std::vector < VkExtensionProperties >
      availableExtensions(extensionCount);

    //3. query the extension details
    vkEnumerateDeviceExtensionProperties(device, nullptr, &
      extensionCount, availableExtensions.data());

    std::set < std::string >
      requiredExtensions(deviceExtensions.begin(),
        deviceExtensions.end());

    for (auto & extension: availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    // 4. verify that swap chain support is adequate
    // it is not enough to check that the swap chain is supported because it may not be compatible with the window surface
    // for now, support is sufficient if there is at least one supported image format and one supported presentation mode given the window surface we have

    bool swapChainAdequate = false;

    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
      !swapChainSupport.presentModes.empty();

    return q.graphicsFamily.has_value() &&
      q.presentFamily.has_value() &&
      requiredExtensions.empty() &&
      swapChainAdequate && deviceFeatures.samplerAnisotropy;

  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, & extensionCount, nullptr);

    std::vector < VkExtensionProperties > availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, & extensionCount, availableExtensions.data());

    std::set < std::string > requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto & extension: availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  struct QueueFamilyIndices {

    // note that it is very likely that the two queues that support graphics and presentation will end up being the same one . but we will continue to treat them distinctly.
    std::optional < uint32_t > graphicsFamily; // INDICES OF VALID QUEUE FAMILIES , optional because we need a way to indicate that a queue family is not available : graphicsFamily.hasValue() returns false if no index was assigned to it.

    std::optional < uint32_t > presentFamily; // need to support window integration

    std::optional < uint32_t > transferFamily; // optional as other queues implicitly support VK_QUEUE_TRANSFER_BIT

  };

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices; // indices of the queue families that satisfy what we need

    // assign index to queue families that could be found

    // query the queue families from physical device :
    // start by query the count
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, & queueFamilyCount, nullptr);

    // allocate the array
    std::vector < VkQueueFamilyProperties > queueFamilies(queueFamilyCount);

    // query the queues
    vkGetPhysicalDeviceQueueFamilyProperties(device, & queueFamilyCount, queueFamilies.data());

    VkBool32 presentSupport = false;

    // we want to find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
    int i = 0;
    for (auto & queueFamily: queueFamilies) {

      // early exit: if we already have a queue that contain has the queue family we need , exit
      // if we already found a queue family that meet our requirement we would have stored its indices , so indices would have a value
      if (indices.graphicsFamily.has_value() && presentSupport && indices.transferFamily.has_value())
        break;

      if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT))  {
        indices.graphicsFamily = i;
      }

      //    if(queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT & !VK_QUEUE_GRAPHICS_BIT){
      //    indices.transferFamily = i;
      //  }

      // function that checks if queue family has presentation capabilities
      // parameters : physical device, queue family index, surface
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &
        presentSupport);

      if (presentSupport) {
        indices.presentFamily = i;
      }

      i++;
    }
    return indices;

  }

  std::vector <
  const char * > getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char ** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions( & glfwExtensionCount);

    std::vector <
      const char * > extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  // check if all the requested layers are available
  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties( & layerCount, nullptr);

    std::vector < VkLayerProperties > availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties( & layerCount, availableLayers.data());

    for (const char * layerName: validationLayers) {
      bool layerFound = false;

      for (const auto & layerProperties: availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  // helper function to read file : read all of the bytes from the specified file and return them in a byte array
  static std::vector < char > readFile(const std::string & filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary); // ate = start reading at end of the file/ binary means read it as binary file

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");

    }

    // reading at end of file allow us to use the read position to determine the size of the file and allocate the buffer
    size_t fileSize = (size_t) file.tellg();
    std::vector < char > buffer(fileSize);

    // seek back to the begginig of the file and read it all at once
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
  }

    
    
  VkBuffer stagingBuffer; // buffer where we will copy the pixels of the loaded image
  VkDeviceMemory stagingBufferMemory; // the memory associated with the staguing buffer

  VkImage textureImage; // the shader could have accessed the pixel values from a buffer but it is faster to use image object (can use 2d coord)
  VkDeviceMemory textureImageMemory;

  
  // helper function to handle layout transition :
  // perform layout transition : use image memory barrier.
  // pipleine barrier is used to synchronize access to ressourcea , but can also be used to transition image layouts and transfer queue family ownership when the sharing mode is set to exclusive.
  void transitionImageLayout(VkImage image, VkFormat format,
    VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // Note that there is an equivalent to this for buffers
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // layout transitions :
    barrier.oldLayout = oldLayout; // if you don t caer about the old layout , you can set it to VK_IMAGE_LAYOUT_UNDEFINED
    barrier.newLayout = newLayout;
    // this is only relevent when you are using the barrier to transfer ownership between two queue for ressource set to be exclusive. In that case set these to the indices of the queue families.
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    // specify the image affected and the specific image part. Ou rimage is not an array and doesn t have mipmap levels. so only one level and layer are specified.

    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    // barriers are primarly used for synchronization purposes, so you must specify whjich type of opeartions that involves the ressource (here the image) must happen before the barrier and which operations that involves the ressource must wait for the barrier.
    // We need to do this despite using vkQueueWaitIdle to manually synchrionize .
    // the values depend on the old and layout layout.
    // if the old layout is undefined and we want the new layout to be the transfer destination : tranfer writes that don t need to wait or anything
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout ==
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0; // since the writes don t have to wait for anything, the we can specify an empty access mask.
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; // earliest possible stage in the piepliene
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // noot a Real stage but a pseudo-stage.
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
      newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // the image will be written in the same piepine stage and be read by the fragment shader : spcify fragment shader pipeline stage
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    // submit the pipeline barrier
    vkCmdPipelineBarrier(
      commandBuffer,
      sourceStage, // specify in which pipeline stage the operation that should happen before the barrier occur. The pipeline stages that you are allowed to specify before and after the barrier depend on how you use the ressource (imagfe) before and after the barrier
      destinationStage, //  specify in which pipeline stage the operation that should happen after the barrier occur. ex: you are going to read from uniform after the barrier (usage is VK_ACCESS_UNIFORM_READ_BIT), so you should specify the earliest shader that will read from the uniform as pipeine stage , ex: VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT.
      0, // either 0 or VK_DEPENDENCY_BY_REGION_BIT : turns the barrier into a per-region condition. ex of usage : the implemnetation is allowed to already begin reading from the parts of a ressource that were written do far.
      // last 3 pairs foparameters reference arrays of piepeline barriers of the three available types : memory barriers, buffer memory barriers, image memeory barriers (we are using this one).
      0, nullptr,
      0, nullptr,
      1, & barrier
    );
    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    // this is gpu operations , so it needs to be recorded in command  buffer to be submitted to GPU : done by the helper function : cerates a temporary command buffer , tighed to the command pool with graphics trasnfer capabilities
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // specify the region of the buffer to copy (source) to which part of the image (destination )
    VkBufferImageCopy region {};
    region.bufferOffset = 0; // the byte offset in the buffer at which the pixel values start
    //how the pixels are laid on in memory  : 0 means that the pixels are simply tightly packed. meaning tthat we so not have padding bytes between rows of image.
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    // indicate to which part of thw image we want to copy the pixels.
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0,0,0};
    region.imageExtent = {width,height,1};

    vkCmdCopyBufferToImage(
      commandBuffer,
      buffer,
      image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // indicates the layout the image is currently using (assuming the layout ahs already transitionned to the most optimal layout for transfer of pixels )
      1, &
      region // note that it is possible to many copies from buffer to the image in one operation : specify array of VkBufferImageCopy
    );

    endSingleTimeCommands(commandBuffer);

  }
    
  // helper functions for recording and executing command buffer
  VkCommandBuffer beginSingleTimeCommands() {
    // 1. allocate a temporary command buffer to handle the memory transfer operation
    // because the graphics queue is implicitly a transfer queue as well, we can use the same command pool to allocate this command buffer

    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, & allocInfo, & commandBuffer);

    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, & beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    //4. Submit the command buffer :
    // queue submission and synchronization is done by filling VkSubmitInfo
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = & commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, & submitInfo, VK_NULL_HANDLE);

    // 5.we want to submit the queue immediately
    // either use fence and wait with vkWaitForFences or wait for the transfer to be done , we know that the transfer is done when the transfer queue (here it is the graphics queue that also transfers ) become idle.
    vkQueueWaitIdle(graphicsQueue);

    //6. cleanup the command buffer
    vkFreeCommandBuffers(device, commandPool, 1, & commandBuffer);
  }

  void createImage(uint32_t width, uint32_t height, VkFormat format,
    VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties, VkImage & image,
    VkDeviceMemory & imageMemory) {
    //5. Create an image, whre we will copy the data from the staging buffer to. the shader will then access this image.
    // the shader could have accessed a buffer , but uit wil be faster to access image , as wit s able to use 2d coord
    // creating image requires filling up VkImageInfo:
    VkImageCreateInfo imageInfo {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D; // tells Vulkan with what kind of coordinate system the texel in the image are going to be addressed (texel is a pixel within the image). ex: 1d can store array of data, gradient / 2d can be used for textures, 3D : can be used to store voxal volumes ... etc.
    imageInfo.extent.width = width; // dimensions : how many pixels on each axis
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1; // we are in 2d image, there is 1 pixel on z axis.
    imageInfo.mipLevels = 1; // not useing mipmaping
    imageInfo.arrayLayers = 1;
    imageInfo.format = format; // image format : use the same format between the texel and the pixels in the buffer we are copying from otherwise you will not be able to copy from one to another
    imageInfo.tiling = tiling; // specify how texels are laid down: either laid down in row major order like our pixel array / or texels are laid down in an implementation defined order for optimal access
    // the tiling (unlike the layout of the image) cannot be changed later.
    // this depends on how you will access the texels in the iage : To access texel directly in the memory of the image then use VK_IMAGE_TILING_LINEAR. In our case, we wil be using staging buffer so it won t be necessary. we are using VK_IMAGE_TILING_OPTIMAL for efficient access from shader.
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // 2 possibilities : VK_IMAGE_LAYOUT_UNDEFINED (not usable by the GPU and the very first transition will discard the texels )  or  VK_IMAGE_LAYOUT_PREINITIALIZED (not usable by the GPU , but the first transition will preserve the texels).
    // There is certain situations where it s necessary for the texels to be preserved during the first transition : ex: if you want to use image as staging image (need VK_IMAGE_LAYOUT_PREINITIALIZED in that case to be able to access the tile directly in memory), in that case, you would need to upload texel data to it and then the image to be transfer source without losing data. In our case, the image wil be a transfer destination and then cipy texel data to it from the staging buffer object so we don t care about losing data.
    imageInfo.usage = usage; // image used as destination for the buffer copy
    //VK_IMAGE_USAGE_SAMPLED_BIT; // we want the image to be accessible from the the shader to color the meshes.
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // image will only be used by 1 quueue family : the graphics one that supports graphics transfer
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; // related to multi sampling
    imageInfo.flags = 0; // Optional : realted to sparce image : images where only certain regions are actually backed by memory. ex: 3 d texture for voxel terrain, you could use that to avoid allocating memory ofor air texture.

    // create the image object
    if (vkCreateImage(device, & imageInfo, nullptr, & image) !=
      VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    // 6. Allocate memory for the image :
    // same way we allocate memory for the buffer :

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, & memRequirements);

    VkMemoryAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    //get the requirements of the memeory to use : requirement associated to the type buffer , image .. and the ones assocuated with the application.
    allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits,
        properties);

    if (vkAllocateMemory(device, & allocInfo, nullptr, &
        textureImageMemory) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    // bind the memeory allocated to the image :
    vkBindImageMemory(device, image, imageMemory, 0);

  }

 

  // Helper function to create image views
  VkImageView createImageView(VkImage image, VkFormat format) {
    VkImageViewCreateInfo viewInfo {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;

    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, & viewInfo, nullptr, & imageView) !=
      VK_SUCCESS) {
      throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
  }

    // helper function to create buffers
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
      VkMemoryPropertyFlags properties, VkBuffer & buffer,
      VkDeviceMemory & bufferMemory)

    {
      // 1. creating a buffer requires to fill VkBufferCreateInfo
      VkBufferCreateInfo bufferInfo {};
      bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferInfo.size = size;
      bufferInfo.usage = usage;
      bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      if (vkCreateBuffer(device, & bufferInfo, nullptr, & buffer) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer by helper function!");
      }

      //2 . allocate memeory to it
      // now buffer is created but no memory has been assigned to it
      //2.a :  : query the memory requirements.
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(device, buffer, & memRequirements);

      // 2.b allocate the memory, now that we are able to determine the required memeory according the  buffer requirement and suits the application we are using it in.
      //fill VkMemoryAllocateInfo :
      VkMemoryAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize = memRequirements.size;
      allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

      //2.c pass the info struct to the creation of the buffer :
      if (vkAllocateMemory(device, & allocInfo, nullptr, &
          bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory!");
      }

      // 2.4 associate the vertex buffer handler to the memory we just allocated
      vkBindBufferMemory(device, buffer, bufferMemory, 0); // the last paramter: the offset within the region of memory
    }
      
      
      // memory transfer operations are excecuted using command buffers
      void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
          // 1. allocate a temporary command buffer to handle the memory transfer operation
          // because the graphics queue is implicitly a transfer queue as well, we can use the same command pool to allocate this command buffer
          VkCommandBufferAllocateInfo allocInfo{};
          allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
          
          allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
          allocInfo.commandPool = commandPool;
          allocInfo.commandBufferCount = 1;
          
          VkCommandBuffer commandBuffer;
          vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
          
          
          //2.start recording the command buffer:
          VkCommandBufferBeginInfo beginInfo{};
          beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
          beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // we are only going to use the command buffer once and wait until the operation has finished executing.
      
           vkBeginCommandBuffer(commandBuffer, &beginInfo);
          
          //3. transfer the content of the staging buffer to the vertex buffer :
          // define the region to transfer
          VkBufferCopy copyRegion{};
          copyRegion.srcOffset = 0; // Optional
          copyRegion.dstOffset = 0; // Optional
          copyRegion.size = size;
          // the command that do the transfer : takes the src , dst and the region to transfer
          
          vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
          
          vkEndCommandBuffer(commandBuffer);
          
          
          //4. Submit the command buffer :
          // queue submission and synchronization is done by filling VkSubmitInfo
          VkSubmitInfo submitInfo{};
          submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          submitInfo.commandBufferCount = 1;
          submitInfo.pCommandBuffers = &commandBuffer;
          
          vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
          
          // 5.we want to submit the queue immediately
          // either use fence and wait with vkWaitForFences or wait for the transfer to be done , we know that the transfer is done when the transfer queue (here it is the graphics queue that also transfers ) become idle.
          vkQueueWaitIdle(graphicsQueue);
          
          //6. cleanup the command buffer
          vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
          
      }
    
    
    
    // Helper funciton to find the approproate memeory type accoriding to the buffer we are creating and the use we intend :
    // VkMemoryRequirements has 3 fields :
    /* memory types deffers from gpu to another, need to combine our requiements for the buffer with the requirements fomr the system
     -size : required amount of memory in bytes : may differ from bufferInfo.size
     - alignment : offset in bytes where the buffer begins in the allocated region of memory
     -memory type buts : bit field of the memory types that are suitable for the buffer */
    // find the right memory type
    // find the right memeory type
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) // Property = define special features of the memory (ex: able to map it, so we can write on it from CPU)
    {
      // query info about the available memory
      VkPhysicalDeviceMemoryProperties memProperties; // this struct contains 2 arrays : memoryTypes and mempryHeaps
      vkGetPhysicalDeviceMemoryProperties(physicalDevice, & memProperties);

      for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {

        // find the index of suitable memory type by iterating over them and checking if the corresponding bit is set to 1.
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) { // typeFilter specifies the bit field of memory types that are suitable.

          // we are not only interested t fond memory that suits the vertex , we shoudl be able yo write vertex data to that memory
          // use memory Types array : it consists of VkMempryType structs tthat specify the heap and memepry type of each type of memeory. Property = define special features of the memory (ex: able to map it, so we can write on it from CPU)

          return i;
        }
      }
      throw std::runtime_error("failed to find suitable memory type!");
    }

    /* Synchrionization graphics and compute :
     if you do not synchronize correctly, the vertex stage may start drawing (reading from the ssbo ) before the compute shader finished updating the data. Or compute shader may start updating data that is till being read from the vertex shader.
     if you use 2 separate submits to submit the draw and dispatch , we can use semaphores and fenses to ensure that the vertex shader will only start reading the data when the compute shader finished writing.
     
     Note that even thought the submits are one after the other, there is no guarantee that the compute would have fished before the vertex starts reading.
     
     */
    
    
   void recordComputeCommandBuffer(VkCommandBuffer computeCommandBuffers){
        // begin recording commands
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
        if (vkBeginCommandBuffer(computeCommandBuffers, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
        }
        
        // bind compute piepline
        vkCmdBindPipeline(computeCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipeline);
        
        // bind the descriptor set needed for this pipeline
        vkCmdBindDescriptorSets(computeCommandBuffers,
        VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1,
        &descriptorSets[currentFrame], 0, 0);
        
        // Dispatch : dispatch  PARTICLE_COUNT / 256 local work groups in the x, dimension (our input is 1d so only specify in x dimension).
        // we already defined that each compute shader in a workgroup has local size of 256 = every work group will do 256 invokatoons.
        // imagine we have 4096 particles, each workgroup will run 256 compute shader invocation so we will dispatch 16 work group (each running 256 invokations ).
        // if the particle size is dynamic, and cannot be always divided by 256 , you can use gl_GlobalInvocationID at start of compute shader and return from it if the glocal invocation index is greater than the number of particles.
        vkCmdDispatch(computeCommandBuffers, PARTICLE_COUNT / 256, 1, 1);
        
        // end recording commands
        if (vkEndCommandBuffer(computeCommandBuffers) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
        }
        
                                  
    }
    // helper function to update the uniform buffer during the rendering :
    void updateUniformBuffer(uint32_t currentImage) {
      static auto startTime = std::chrono::high_resolution_clock::now();

      auto currentTime = std::chrono::high_resolution_clock::now();

      float time = std::chrono::duration < float, std::chrono::seconds::period > (currentTime -
        startTime).count(); // time per second

      UniformBufferObject ubo {}; // uniform buffer object

      // model transformation : ex: rotation around the Z axis :
      // rotation angle:time * glm::radians(90.0f) : rotate 90 degree o=per second
      // rotation axis : z axis : glm::vec3(0.0f, 0.0f, 1.0f)
      ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

      //view projection : from above 45 degree angle :
      ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), // eye position
        glm::vec3(0.0f, 0.0f, 0.0f), //center position
        glm::vec3(0.0f, 0.0f, 1.0f)); //up axis

      // perspective projection :
      ubo.proj = glm::perspective(glm::radians(45.0f), // perspective projection with 45 degree vertical field of view.
        swapChainExtent.width / (float) swapChainExtent.height,
        // aspect ratio
        0.1f, // near plane
        10.0f); // far plane

      // flip the y because glm was designed for openGL where y coord os the clip coord is inverted
      // flip the sign of the scaling factor of y : (otherwise the image is rendered upside down)
      ubo.proj[1][1] *= -1;

      // copy the data in the uniform buffer object of the current uniform buffer
      memcpy(uniformBuffersMapped[currentImage], & ubo, sizeof(ubo));
    };

  //operations in draw frame are asynchronous, so when we exit the main loop, rendering and oresenting may still be going on.
  void drawFrame() {
    /* Rendering a frame in Vulkan :
     -1. Wait for previous frame to finish : host waits for previous frame to finish
     -2. acquire an image from the swap chain
     -3.  record a command buffer which draws the scene onto that image
     -4. submit the recorded command buffer
     -5. present the swap chain image
     */
    // update the uniform buffer :

    //1. Wait for previous frame to finish
    // takes an array of fences and waits on the host for either any or all of the fences to be signaled before returning
    // VK_TRUE = want to wait for all the fences to finish
    //UINT64_MAX = timeout parameter , here this disables the timeout
    vkWaitForFences(device, 1, & inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // after waiting we need to manually reset the timeout:
    vkResetFences(device, 1, & inFlightFences[currentFrame]);

    //2. acquire an image form the swap chain :
    uint32_t imageIndex;
    // first two parameters : logical device and swap chain from which we want to acquire an image
    // 3rd parameter: time-out in nano sec
    //4rth, 5th params : image available semaphore/ null handler:  two parameters : specify the synchronization objects that are to be signaled when the presentation engine is finished using the image. this is when we can start drawing on it.
    // last param : imageIndex : variable to output the index of the swap chain image that has become available. it is the index of the swap chain image that has become available. this is then  used to pick up the corresponding frame buffer
    vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
      imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, & imageIndex);

    vkWaitForFences(device, 1, &computeInFlightFences[currentFrame],VK_TRUE, UINT64_MAX);
      
    updateUniformBuffer(currentFrame);

    //3. record the command buffer :
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    //4. Submit the command buffer :
    // queue submission and synchronization is done by filling VkSubmitInfo
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // specify which semaphores to wait on before execution begins and in which stages of the pipeline to wait.
    VkSemaphore waitSemaphores[] = {
      imageAvailableSemaphores[currentFrame]
    };
    // stage of the pipeline to wait in : wait with writing color to the image, until the image is available : the stage in the pipeline that writes to the color attachment. Each entry in the waitStages corresponds to the semaphore of same index in the waitSemaphores
    VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    // Which command buffer to actually submit.
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = & commandBuffers[currentFrame];

    // specify which semaphore to signal once the command buffer have finished executing. in our case, when the command buffer has finished executing, signal that the rendering has finished , meaning that the presentation can happen
    // the presentation will have to wait for this semaphore to signal
    VkSemaphore signalSemaphores[] = {
      renderFinishedSemaphores[currentFrame]
    };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    //Submit the command buffer to the graphics queue
    // last paramter: the fence to be signaled when the command buffer finishes execution : allow to know when it is safe for the command buffer to be reused . In the next frame, CPU will wait for this fence to be signaled before overriding the command buffer.
      // when submitted the graphics queue will use the updated data by the compute shader
    if (vkQueueSubmit(graphicsQueue, 1, & submitInfo, inFlightFences[currentFrame]) !=
      VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    //5. Presenting : submit the result back to the swap chain to have it eventually show up on the screen
    VkPresentInfoKHR presentInfo {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    // which semaphore to wait on before presenting :
    // we want to present when the command buffer finishes executing
    // wait for the semaphores that are signaled at the end of command buffer execution
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores; //(here signal semaphore signals that the rendering happened and we are ready to present)

    // specify the swap chain to present images to
    VkSwapchainKHR swapChains[] = {
      swapChain
    };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = & imageIndex; // the index of the image for each swap chain

    presentInfo.pResults = nullptr; // Optional : specify an array VkResult values to check for every individual swap chain if presentation is succesful , not necessary when you have only one swap chain.

    // submit the request to present an image to the swap chain
    vkQueuePresentKHR(presentQueue, & presentInfo);

    // advance the frame :
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // use modulo to ensure that the frame index loops around after every Max_frame_InFLight enqued frames.
  }

   
    // INIT VULKAN FUNCTIONS :
    
  //a. create instance
  void createInstance() {
   
      // validation layer available and enabled ?
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error("validation layers requested, but not available!");
    }

    //1. Optional : fill a struct with some information about the app. the data is technically optional, but may provide useful info the driver for optimization
    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // 2. tell Vulkan driver which global extensions (applied to whole program and not specific device)
    // and validation layers we want to use.
    VkInstanceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = & appInfo;

    uint32_t glfwExtensionCount = 0; // where to store the number of extension in the returned array
    const char ** glfwExtensions;
   
      // specify the desired global extensions
     // here we need to use an extension to interface with the window system. this is provided by glfw
    glfwExtensions = // This function returns an array of names of Vulkan instance extensions required
      //by GLFW for creating Vulkan surfaces for GLFW windows.If successful, the
      // list will always contain `VK_KHR_surface`
      glfwGetRequiredInstanceExtensions( & glfwExtensionCount); // return the extensions required  , which contains surface extension and other WSI extensions . contain platform specific addition to the extensions : VK_KHR_win32
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions; //  An array of ASCII encoded extension names

    // specify the global validation layer to enable
    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
        static_cast < uint32_t > (validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    std::vector <const char * > requiredExtensions;

    for (uint32_t i = 0; i < glfwExtensionCount; i++) {
      requiredExtensions.emplace_back(glfwExtensions[i]);
    }

    requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    requiredExtensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    createInfo.enabledExtensionCount = (uint32_t)
    requiredExtensions.size();
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();
   
    // 3. Create instance
    // parameters : pointer to struct wth creation info / pointer to custom allocator callback / pointer to the variable that stores the handle to the new object

    if (vkCreateInstance( & createInfo, nullptr, & instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }

    // retrieve a list of supported extensions before creating instance to make sure the extension specified is present

    // 1.first we need to know how many extensions are they to create an array to hold them :
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, & extensionCount,
      nullptr);

    // 2.now we have the number of parameters, we can allocate an array to hold the extension details :
    // each VkExtensonProperties struct contains the name and version of an extension.
    std::vector < VkExtensionProperties > extensions(extensionCount);
    
      //3. query the extension details
    vkEnumerateInstanceExtensionProperties(nullptr, & extensionCount,
      extensions.data());

    // list the available extensions
    std::cout << "available extensions:\n";

    for (const auto & extension: extensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }

  }

  //b.
  void createSurface() {
    // PARAMTERS : vkInstance, GLFW window pointer, custom allocator, pointer to VKSurfaceKHR variable
    // this function passes through the vkResult from the relevant platform call (so that we don not deal with platform specific code)
    if (glfwCreateWindowSurface(instance, window, nullptr, & surface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }

    // make sure that your physical device support window system integration
    //  but this is a queue specific feature : need to support PRESENTATION COMMANDS
  }

  //c.
  void pickPhysicalDevice() {
    // list the graphics card available

    // first get the count : only query the count
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, & deviceCount, nullptr);

    // allocate the array where to store them
    std::vector < VkPhysicalDevice > devices(deviceCount);

    // query the data
    vkEnumeratePhysicalDevices(instance, & deviceCount, devices.data());

    std::cout << "GRAPHIC CARDS ";
    std::cout << deviceCount;
    // want to see all what i have
    for (const auto & device: devices) {
      std::cout << device;
    }

    // verify if any of the physical devices meet the requirements
    for (const auto & device: devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }
    if (physicalDevice == VK_NULL_HANDLE) // no device is suitable
    {
      throw std::runtime_error("failed to find a suitable GPU");
    }
  }

  //d.
  void createLogicalDevice() {
    // specify bunch of details in structs .
    // note that the queue are created along with the logical device but we don t havae a handle to interface with them.

    //1. describe the number of queue we want for a single queue family
    // now we only want a queue with graphics capabilities and presentation capabilities
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice); // all queues with a certain requirement we set the physical device supports

    // we need to have multiple vkDeviceQueueCreateInfo structs to create a queue from both families.
    // we will do that by creating a set of all unique queue families that are necessary for the required queue.
    std::vector < VkDeviceQueueCreateInfo > queueCreateInfos;
    std::set < uint32_t > uniqueQueueFamilies = {
      indices.graphicsFamily.value(),
      indices.presentFamily.value()
    }; // set only stores unique elements

    float queuePriority = 1.0f;

    for (uint32_t queueFamily: uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo {}; // the struct to hold the nbr of queues from a single family

      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; // struct type
      queueCreateInfo.queueFamilyIndex = queueFamily; // indices of the queues that have graphics capabilities (according to our check)
      queueCreateInfo.queueCount = 1; // nbr of queue per family of queue

      // 2.you can assign priority to queue which influence the scheduling of command buffer execution
      // this is REQUIRED even when you have only one queue
      queueCreateInfo.pQueuePriorities = & queuePriority;

      queueCreateInfos.push_back(queueCreateInfo);
    }

    // 3.specify the set of device features that we will be using.
    // these are the features that we made sure the device supports
    //VkPhysicalDeviceFeatures deviceFeatures{}; variable created
    if (enableValidationLayers) {

      //  deviceExtensions.emplace_back()(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }

    //4. we have the infos structs required to fill the VKDeviceInfo
    VkDeviceCreateInfo createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; // structure type

    // add pointer to the queue creation info and device feature struct
    createInfo.queueCreateInfoCount =
      static_cast < uint32_t > (queueCreateInfos.size());
    createInfo.pQueueCreateInfos = (queueCreateInfos.data());
    createInfo.pEnabledFeatures = & deviceFeatures; // specified outside, nothing in it for now

    // ignored by new vulkan implementation
    createInfo.enabledExtensionCount = static_cast < uint32_t > (deviceExtensions.size());;
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
        static_cast < uint32_t > (validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

    } else {
      createInfo.enabledLayerCount = 0;

    }

    // 5. instantiate the logical device with call the vkCreateDevice.
    // parameters: physical device , queue / usage info/ optionnal allocation callback/ variable to store the logical device
    if (vkCreateDevice(physicalDevice, & createInfo, nullptr, & device) != VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    // the queues are created along with the logical device, but we don t have a handle for them hyet, need retrieve it and store it :
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &
      graphicsQueue); // graphics queues stores the queues handles

    // we need to do the same for presentation queue
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &
      presentQueue);
      
      vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &computeQueue);

  }

  // e. Find the right settings for the best possible swap chain
  // there are three types to determine
  void createSwapChain() {

    // pass the details of the swap chain : the capabilities of the swapchain
    SwapChainSupportDetails swapChainSupport =
      querySwapChainSupport(physicalDevice);

    // 1. surface format = color depth :
    //  VkSurfaceFormatKHR entries has format and color space members
    // format specify the color channel and type
    // the color space specify if the SRGB color space is supported or not
    //  we want to use SRGB if it is available (more accurate percieved color)
    // need to use srgb color format ex : VK_FORMAT_B8G8R8A8_SRGB
    VkSurfaceFormatKHR surfaceFormat =
      chooseSwapSurfaceFormat(swapChainSupport.formats);

    //2. Presentation mode : the most important setting for the swap chain
    // represents the actual conditions for showing images in the screen
    VkPresentModeKHR presentMode =
      chooseSwapPresentMode(swapChainSupport.presentModes);

    // 3.swap extent : resolution of the swap chain images
    // almost always equal to the resolution of the window we are draweing to in pixels/

    // vulkan tells us to match the resolution of the window by setting the width and hight in the currentExtent
    // capabilities has member current extent that uses pixel
    VkExtent2D extent =
      chooseSwapExtent(swapChainSupport.capabilities);

    // 4. Decide how many images we would like to have in the swap chain.
    // for now, we specify the minimum that it requires to function +1
    // sticking to the min means we have sometimes to wait for the driver to complete internal operations before we can aquire
    // another image to render to .
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    // 5. make sure to not exceed the maximum number of images
    // 0 is special character meaning there is no maximum
    if ((swapChainSupport.capabilities.maxImageCount > 0) && (imageCount > swapChainSupport.capabilities.maxImageCount)) {
      imageCount = swapChainSupport.capabilities.maxImageCount;

    };

    //6. create the swap chain : requires to fill in a large structure
    VkSwapchainCreateInfoKHR createInfo {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface; // which surface (cross platform abstraction over window to render to)the swap chain should be tied to

    //7.  specify the swap chain images (the one that we determined earlier):
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1; // specify the amount of layers each image consists of. always 1 unless you are developing a sterostopical 3d app
    // specifies what kind of operations we will use the image in the swap chain from.
    // for now, we will render directly to them , so they are used as color attachment.
    // it is possible that you want to render images to separate image than perform operations like post processing.
    // them is a memory transfer the rendered image to the swap chain image
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // specify how to handle swap chain images that are to be used with multiple queue families.
    // in our case, this is the case if the presentation and graphics queues are not the same:
    // we want to draw in the images in the swap chain from the graphics queue and submit them on the presentation queue.
    // 2 ways to handle images that are accessed from multiple queues:
    // either explicit transfer of ownership of the image, or without : concurrent mode (in that case, specify the sharing entries first.)
    // if the two queues are the same, stick to exclusive mode as you don t have 2 distinct queues to specify
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // you can add a transformation to the image if supported
    // if you don t want, just assign the current transform to it
    createInfo.preTransform =
      swapChainSupport.capabilities.currentTransform;

    // specify if the alpha channel should be used for blending with other windows in the window system.
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // ignore the alpha channel

    // specify the present mode we determined earlier
    createInfo.presentMode = presentMode;

    // clipped true means that we don t care about the color of pixels that are obscured (ex: another window is in front of them)
    // better performance when enabled
    createInfo.clipped = VK_TRUE;

    // for now, we only have one swap chain, this is supposed to be reference to swapchain in case it needs to be recreated from scratch
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    // 8. Create the swap chain :
    // parameters : logical device / swap chain creation info / custom allocator / pointer to stotre swap chain
    if (vkCreateSwapchainKHR(device, & createInfo, nullptr, & swapChain) !=
      VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    // 9. the implementation of swap chain creation has also created the images
    // now we need to retrieve the handles to these images
    // retrieve the array of handles : 2 steps to retrieve the count then use it to retrieve the data
    // the images will be automatically destroyed
    vkGetSwapchainImagesKHR(device, swapChain, & imageCount, nullptr);
    swapChainImages.resize(imageCount); // resize the vector according to the number of images

    vkGetSwapchainImagesKHR(device, swapChain, & imageCount,
      swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    // now we have a set of images that be drawn to and that can be presented to the window
  };

  VkDescriptorSetLayout  ComputeDescriptorSetLayout;
  
    // f.
  void createImageViews() {
    //1. resize the list of image views to fit all the image views we will be creating
    swapChainImageViews.resize(swapChainImages.size());

    //2. loop over all the swap chain images and create an image view for each of them
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      // specify the parameters needed to create the image view
      // populate struct VkImageViewCreateInfo
      VkImageViewCreateInfo createInfo {};

      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];

      // specify how the image data should be interpreted
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // treat image as 1d / 2d / cube map ...etc
      createInfo.format = swapChainImageFormat;

      //component field allows you to swizzle the color channel around.
      // the following is to stick with the default mapping
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      // subresourceRange describes what the image's purpose is and which aprt of the image should be accessed
      // for now, our image will be used as color targets without any mipmapping or multiple layers
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      //  create the image view :
      if (vkCreateImageView(device, & createInfo, nullptr, &
          swapChainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");

      }
    }

  };

    //g. create render pass :
    // render pass object wrap all the info we need to tell vulkan before drawing
    void createRenderPass() {

      //1. attachment description
      // one color buffer represented by one of the images from the swap chain
      VkAttachmentDescription colorAttachment {};
      colorAttachment.format = swapChainImageFormat; // this format should match the format of the swap chain image
      colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // if we do not do multi-sampling , only one sample is enough

      // loadOp and storeOp determine what to do with the data in the attachement before and after rendering
      // they apply to color and depth data. there is stencilLoadOp /stencilStoreOp apply to stencil data
      colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear the values at constant to start : here clear the framebuffer to black before drawing new frame
      colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // render contents will be stored in memory and can be read later (the other option is that the content of the framebuffer is undefined after the rendering operation)..we want our triangle to stay on the screen, use the store.

      // images need to be transitioned to specific layouts that are suitable for the operation that they are going to be involved in next
      // the layout of the pixels in the memory can change based on what you are trying to do woth the image
      // ex of common layouts : layout for image to be used in swapchain/ layout for image used as color attachment / layout for images to be used as destination for a memory copy operation
      colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // initial layout ; the layout the image has before the rendering begins. here we don t care about about the previous layout the image was in. the content of the image are not guaranteed to be preserved but we are clearing it anyway .
      colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // final layout : specifies the layout to automatically transition to when the render pass finishes. // here we want it to be presented in the swap chain

      //2. Subpasses and attachement reference
      // A single render pass can consist of multiple subpasses: subsequent rendering operations that depend on content of framebuffers in previous passes
      // ex: sequence of post processing effects applied one after the other
      // when a sequence of rendering operation are grouped in a same render pass, vulkan is able to reorder them while conserving memory bandwidth (rate of data transfer/ memory usage) for possibly better performance.
      // in our case we have one single subpass

      // every subpass references one or more of the attachments that we described (VkAttachmentDescription)
      // these references are VkAttachmentReference
      VkAttachmentReference colorAttachmentRef {};
      colorAttachmentRef.attachment = 0; // specifies which attachment to reference by its index in the attachment descriptions arrays.
      // our array consists of only one VkAttachmentDescription so its index is 0.

      colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // specifies which layout we would like the attachment to have during a subpass that uses this reference. Vulkan will automatically transition the attachment to this layout when the subpass starts.
      // in our case, we intend to use the attachment to function as color buffer.
      // the transition of layout is controlled by subpass dependencies. even though we have one subpass, the operations right before and right after the subpass also counts as implicit subpasses.

      // the subpass is described using VkSubpassDescription
      // need to specify the type of the subpass (compute/ graphics .. )
      VkSubpassDescription subpass {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

      // specify the reference to the color attachment
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = & colorAttachmentRef; // other attachments types are input attachments (input to shader), resolve attachments (used for multisampling) , depth stencil attachment, preserve attachment (attachment s that are not used by this subpass but for which the data must be preserved)

      //The index of the attachment in this array is directly referenced from the fragment
      //shader with the layout(location = 0)out vec4 outColor directive

      //3. create the render pass itself
      // need to fill struct VkRenderPassCreateInfo : specify the array of attachments and subpasses : VkAttachmentReference
      VkRenderPassCreateInfo renderPassInfo {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      renderPassInfo.attachmentCount = 1;
      renderPassInfo.pAttachments = & colorAttachment;
      renderPassInfo.subpassCount = 1;
      renderPassInfo.pSubpasses = & subpass;

      if (vkCreateRenderPass(device, & renderPassInfo, nullptr, &
          renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
      }

      // subpass dependencies : responsible of the layout transition between subpasses.
      // currently we have one one subpass, the operations that happens right before and right after count as implicit subpases.
      // 2 dependencies that care of the transition at the start of the render pass and at the end of the render pass. The one that happens at the end of the render passs assumes that the transition occurs at the start of the pipeline, which is not the case.
      // we should make the render pass wait for the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT ( the stage in the pipeline that writes to the color attachment.)
      VkSubpassDependency dependency {};
      // specify the indices of the dependency and rhe dependent subpass.
      // VK_SUBPASS_EXTERNAL: refer to the implicit subpass before or after the render pass depending whether it s in src or dst.
      // indice 0 refer to our subpass (the first and only one). the indice put in dst should always be higher than the one in src unless one of them is VK_SUBPASS_EXTERNAL. this is to avoid cycles in the dependency graph.
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;

      // These two fields specify the operations to wait on and the stages in which these operations occur.
      dependency.srcStageMask = // wait for the swap chain to finish reading from the image before we can access it : wait for the color attachment output stage
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;

      // The operation that should wait on this are in the color attachment stage and involve writing of the color attachment.These setting will prevent the transition from happening until it s actually necesary: when we want to start writing colors in it.
      dependency.dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

      // specify the array of dependencies.
      renderPassInfo.dependencyCount = 1;
      renderPassInfo.pDependencies = & dependency;

    }

  // h. The descriptor set layout (describe actual type of buffer or image) needs to be ready for the pipeline creation
  // each descriptor set layout needs to be described here, and all the descriptor set layouts are bound to single object descriptorSetLayout
  void createDescriptorSetLayout() {

    // every binding needs to be described using : VkDescriptorSetLayoutBinding
    // this is later referenced int the VkDescriptorSetLayoutCreateInfo

    // the uniform buffer descriptor set layout :
    VkDescriptorSetLayoutBinding uboLayoutBinding {};
    // the binding used in the shader
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // the type of descriptor , here uniform buffer object ( UBO )
    uboLayoutBinding.descriptorCount = 1; // number of values in the array (in our case, we have 1 single uniform buffer object (our struct) )
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // the shader stage in which the descriptor is going to be referenced. can be VK_SHADER_STAGE_ALL_GRAPHICS
    uboLayoutBinding.pImmutableSamplers = nullptr; // Optional : only relevant for image sampling

    // sampler descriptor set layout :
//    VkDescriptorSetLayoutBinding samplerLayoutBinding {};
//    samplerLayoutBinding.binding = 1;
//    samplerLayoutBinding.descriptorCount = 1;
//    samplerLayoutBinding.descriptorType =
//      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
//    samplerLayoutBinding.pImmutableSamplers = nullptr;
//    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // where we intend to use this descriptor : it is in the fragment shader that the color will be determmined . It s possible that we want to use the texture sampling in the vertex shader (ex : to dynamically deform a grid of vertices by a heightmap ).

      
      // We have 2 layout binding for the ssbo , even if we only render a single particle system. This is b/c the particle positions are updated frame by frame based on delta time. This means that each frame needs to know about the last frame's particle position so that it can update them with a new delta time and write them in the new SSBO. cpmpute shader needs to have access to the last and current frame's ssbo. so we pass both to the compute shader the descriptor setup
      
      // shader storage buffer descriptor set layout
      VkDescriptorSetLayoutBinding shaderStorageBufferLayoutBinding {};
      shaderStorageBufferLayoutBinding.binding = 1;
      shaderStorageBufferLayoutBinding.descriptorCount = 1;
      shaderStorageBufferLayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      shaderStorageBufferLayoutBinding.pImmutableSamplers = nullptr;
      shaderStorageBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      
      // shader storage buffer descriptor set layout
      VkDescriptorSetLayoutBinding shaderStorageBufferLayoutBinding2 {};
      shaderStorageBufferLayoutBinding2.binding = 2;
      shaderStorageBufferLayoutBinding2.descriptorCount = 1;
      shaderStorageBufferLayoutBinding2.descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      shaderStorageBufferLayoutBinding2.pImmutableSamplers = nullptr;
      shaderStorageBufferLayoutBinding2.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;


      
      
    // the descriptor layouts to bind
    std::array < VkDescriptorSetLayoutBinding, 1 > bindings = {
         uboLayoutBinding,
        //samplerLayoutBinding,

        
    };
      
      std::array < VkDescriptorSetLayoutBinding, 3 > bindings2 = {
          uboLayoutBinding,
          shaderStorageBufferLayoutBinding,
          shaderStorageBufferLayoutBinding2
          
      };

    // crate the descriptor set layout : use vkCreateDescriptorSetLayout and pass to it VkDescriptorSetLayoutCreateInfo
    //fill VkDescriptorSetLayoutCreateInfo
    VkDescriptorSetLayoutCreateInfo layoutInfo {};
    layoutInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast < uint32_t > (bindings.size());
    layoutInfo.pBindings = bindings.data();

    // all the descriptors binding are combined into one object : descriptorSetLayout
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, & descriptorSetLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
      
      
      VkDescriptorSetLayoutCreateInfo layoutInfo2 {};
      layoutInfo2.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      layoutInfo2.bindingCount = static_cast < uint32_t > (bindings2.size());
      layoutInfo2.pBindings = bindings2.data();

      // all the descriptors binding are combined into one object : descriptorSetLayout
      if (vkCreateDescriptorSetLayout(device, &layoutInfo2, nullptr, & ComputeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute descriptor set layout!");
      }

  };

  // <summary>
  // i. create the pipeline
  // </summary>
  void createGraphicsPipeline() {

    //1. load the shaders :
    // read the shaders :
    auto vertShaderCode = readFile("/Users/sanajallouli/Downloads/CODE PROJECTS SANA/VulkanProject/VulkanProject/shader4.vert.spv");
    auto fragShaderCode = readFile("/Users/sanajallouli/Downloads/CODE PROJECTS SANA/VulkanProject/VulkanProject/shader4.frag.spv");
    auto computeShaderCode = readFile("/Users/sanajallouli/Downloads/CODE PROJECTS SANA/VulkanProject/VulkanProject/shader4.comp.spv");

    //2. create shader module : before we pass the code to the pipeline, we need to wrap it into a VkShaderModule

    VkShaderModule vertShaderModule =
      createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule =
      createShaderModule(fragShaderCode);
      VkShaderModule computeShaderModule =
        createShaderModule(computeShaderCode);


    // the compilation on liking of the SPIRV byte code to machine code for execution by the CPU only happens when the graphics pipeline is created

    // 3. to use the shaders we will need to assign them to a specific pipeline stage
    // specify this by populating the appropriate struct : VkPipelineShaderStageCreateInfo

    // a. fill the structure for the vertex shader
    VkPipelineShaderStageCreateInfo vertShaderStageInfo {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // tell vulkan in which pipeline stage the shader is going to be used. There is an enum for each programmable stage of the pipeline (ex: tessellation, vertex, fragment shaders)
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;

    // specify the shader module containing the code
    vertShaderStageInfo.module = vertShaderModule;
    // specify the function to invoke = entry point to the shader
    // this allows us to combine shaders into one shader module and specify different entry point to differentiate their behaviour
    vertShaderStageInfo.pName = "main";
    // specify values for shader constants can be specified here as well : pSpecializationInfo

    // b. specify the structure for fragment shader
    VkPipelineShaderStageCreateInfo fragShaderStageInfo {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // tell vulkan in which pipeline stage the shader is going to be used. There is an enum for each programmable stage of the pipeline (ex: tessellation, vertex, fragment shaders)
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;

    // specify the shader module containing the code
    fragShaderStageInfo.module = fragShaderModule;
    // specify the function to invoke = entry point to the shader
    // this allows us to combine shaders into one shader module and specify different entry point to differentiate their behaviour
    fragShaderStageInfo.pName = "main";
    // specify values for shader constants can be specified here as well : pSpecializationInfo

    
      
    // d. create an array that contains the two struct info of the vertex and fragment shader
    VkPipelineShaderStageCreateInfo shaderStages[] = {
      vertShaderStageInfo,
      fragShaderStageInfo
    };

    // vulkan requires us to be explicit about most of the states of the graphics pipeline (there is no default for them )

    //4. most of the states have to be baked into the graphics pipeline, and cannot be changed without recreating the pipeline at draw time
    // some properties can still be changed.
    // if you use dynamic state , you specify the dynamic properties in the appropriate struct
    // these properties will be ignored and you will need to specify them at draw time.

    // specify the properties that will be mutable
    std::vector < VkDynamicState > dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState {};
    dynamicState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount =
      static_cast < uint32_t > (dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    //a vertex input state
    // . specify the format of the vertex data that will be passed to the vertex shader
    // the description of the format is to be filled in the appropriate structure :VkPipelineVertexInputStateCreateInfo
    // the description specify the binding and the attribute description
    // - the binding : the spacing between data and wether the data is per-vertex or per-instance
    // - the attribute descriptions : type of attributes passed to the vertex shader , which binding to load them from and at which offset

    // define the appropriate structure to fill
    VkPipelineVertexInputStateCreateInfo vertexInputInfo {};
    vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // for now , we are hardcoding the data inside the shader
    // we need to specify that there will be no vertex data to load

    // pVertexBindingDescriptions and pVertexAttributeDescriptions point to an array of structs that specify the details of binding and attribute for the vertex data
      auto bindingDescription = Particle::getBindingDescription();
       auto attributeDescriptions = Particle::getAttributeDescriptions();
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = & bindingDescription; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast < uint32_t > (attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    //b. input assembly
    // VkPipelineInputAssemblyStateCreateInfo : structure that describe :
    // - the kind of geometry that will be drawn from the vertices : the topology member
    // - specify if the primitive restart should be enabled : if set to true it is possible t break up lines and triangles by using special indexes
    //
    // the topology can be : points from vertices, line from every 2 vertices wihtout reuse,
    // the end vertex of every line is used as start vertex for the next line , triangle from every 3 vertices without reuse,
    // the second and third vertex of every triangle are used as first two vertices of the next triangle
    //
    //normally the vertex are loaded from the vertex buffer in sequential order ,
    // but you use an ELEMENT BUFFER you can specify the indices yourself
    // this allows for optimization to reuse vertices

    VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
    inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // c. viewport and scissors :
    // viewport : describe the transformation from the image to the framebuffer
    // scissor rectangle : define in which regions pixels will actually be stored
    // if the scissor rectangle is smaller than the viewport , some pixels will be discarded .
    // if it is larger than the viewport, all the image is rendered but shrinked to enter the size of the viewport
    //VkRect2D scissor{};
    //    scissor.offset = { 0, 0 };
    //scissor.extent = swapChainExtent;
    // // and fill  VkPipelineViewportStateCreateInfo :
    // 1 VkPipelineViewportStateCreateInfo viewportState{};
    // viewportState.sType =
    //VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    // viewportState.viewportCount = 1;
    // viewportState.pViewports = &viewport;
    //viewportState.scissorCount = 1;
    // viewportState.pScissors = &scissor;
    // in our case, they will be made dynamic
    // we already enabled them
    // and flled out the necessary data
    // now need to specify their count
    VkPipelineViewportStateCreateInfo viewportState {};
    viewportState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    // the actual viewport and scissor will be set a drawing time

    // d. rasterizer : discretize the primitive into fragments : pixel elements that fill the framebuffer
    // // it also performs depth testing, face culling, scissor test
    // it can be configured to output fragments that fill the entire polygon or just the edges
    // specify the configuration by filling VkPipelineRasterizationStateCreateInfo
    VkPipelineRasterizationStateCreateInfo rasterizer {};
    rasterizer.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE; // if true : the fragments that are beyond the near and far palnes are clamped to them (not discarded )
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // if true : the geometry never passes through rasterizer = disables output to the frame-buffer
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // how the fragments are generated for geometry : full. line or point

    // Using any mode other than fill requires enabling a GPU feature.
    rasterizer.lineWidth = 1.0f; // if you put something thicker than 1
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // type of face culling o use
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // specify the vertex order for faces to be considered front facing , we are using counter clockwise beucase we fliped the y .

    // rasterizer can alter the depth values by adding a constant value or biasing them based on a fragment's slope
    // used for SHADOW MAPPING
    rasterizer.depthBiasEnable = VK_FALSE; // not enabled in our case.
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

    // e. Multi-sampling : one of the ways to perform antialiasing
    // it combines the fragment shader results of multiple polygons that rasterize (was discretized from the polygon to the pixel) to the same pixel
    // mainly occurs along edges where the most noticeable aliasing artifact occur
    // multi sampling is less expensive than other techniques because in the cases where we have only one polygons rasterized to a pixel, the fragment shader is run only once. This is cheaper than other anti aliasing methods (ex: rendering to a higher resolution then downscaling)

    VkPipelineMultisampleStateCreateInfo multisampling {}; // struct to fill to configure multi-sampling
    multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE; // enable / disable
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // optional

    // f. Depth and Stencil testong : in case you are using depth and/or stencil buffer
    // configure it by filling VkPipelineDepthStencilStateCreateInfo struct

    // g. Color Blending : when the fragment shader returns a color. it needs to be combined with the color that is already in the frame buffer.
    // color blending mixes the fragments that maps to the same pixel in the frame buffer ( ex: drawing many objects in front of each others ? ) (fragments can override each other/ add up or be mixed)
    // you can either mix the two colors. or use bitwise operation.
    // To configure color blending we choose from these 2 methods
    // g.1 : VkPipelineColorBlendAttachmentState : contains the configuration per attached framebuffer
    // g.2 : VkPipelineColorBlendStateCreateInfo : contains the global configuration of the color blending : bitwise combination
    // if none is enabled, the fragment color will be written to the frame buffer as is.
    // g.1 per framebuffer struct
    /*
     if (blendEnable) { // common use of color blending is implementing alpha blending
     finalColor.rgb = (srcColorBlendFactor * newColor.rgb)
     <colorBlendOp> (dstColorBlendFactor * oldColor.rgb);
     finalColor.a = (srcAlphaBlendFactor * newColor.a) <alphaBlendOp>
    (dstAlphaBlendFactor * oldColor.a);
     } else { // if disabled, the new color is passed unchanged
     finalColor = newColor;
     finalColor = finalColor & colorWriteMask;
     }*/
    VkPipelineColorBlendAttachmentState colorBlendAttachment {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE; // if false, it means that the new color from the fragment shader is passed unmodified
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; //Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; //Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; //Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; //Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    // use of color blending : ex : alpha blending : we want the new color to blend with the old color based on its opacity
    // finalColor.rgb = newAlpha * newColor + (1 - newAlpha) * oldColor;
    // finalColor.a = newAlpha.a;
    // its implementation is as follows :
    /*colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor =
    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
     colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    */

    // g2.global configuration of the color blending : references the array of structures for all the framebuffers
    // this allows you to set blend constants you are able to use as blend factors in the calculation defined above,
    VkPipelineColorBlendStateCreateInfo colorBlending {};
    colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional : the bitwise combination that will be done
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = & colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    // H. Pipeline Layout : define uniform variables and push constants
    // uniforms needs to be declared when creating the pipeline by crating pipeline layout object.
    // this structure also specify the push constants
    VkPipelineLayoutCreateInfo pipelineLayoutInfo {};
    pipelineLayoutInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    //tell vulkan which descriptors the shader will be using
    // it is possible to bind multiple descriptor set simultaneously by specifying descriptor layout for each descriptor set.
    // to reference a specific descriptor set , we would specofy the set :  layout(set = 0, binding = 0)
    pipelineLayoutInfo.setLayoutCount = 1; // Optional
    pipelineLayoutInfo.pSetLayouts = & descriptorSetLayout; // Optional
    // push constants
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(device, & pipelineLayoutInfo, nullptr, & pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    // I. Create the graphics pipeline ; fill the VkGraphicsPipelineCreateInfo
    VkGraphicsPipelineCreateInfo pipelineInfo {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    //i.2 reference the array VkPipelineShaderStageCreateInfo : the shader modules that define the functionality of the
    //programmable stages of the graphics pipeline
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;

    //i.3 referece all the structures describing the fixed function stage : like input assembly, rasterizer, viewport and color blending
    pipelineInfo.pVertexInputState = & vertexInputInfo;
    pipelineInfo.pInputAssemblyState = & inputAssembly;
    pipelineInfo.pViewportState = & viewportState;
    pipelineInfo.pRasterizationState = & rasterizer;
    pipelineInfo.pMultisampleState = & multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // Optional
    pipelineInfo.pColorBlendState = & colorBlending;
    pipelineInfo.pDynamicState = & dynamicState;

    // i.4 pipeline layout : the uniform and push values referenced by the shader that can be updated at draw time
    pipelineInfo.layout = pipelineLayout;

    //i.5 reference to the render pass and the index of the sub pass where the graphics pipeline will be used.
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    // Vulkan allows you to create new graphics pipleines by deriving from other pipelines.
    // it is less expensive to set up pipelines wheb they have much functionalities in common with existing pipeline.
    // also swiching between pipelines from the same parent is also quicker
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional : specify the handle of an existing pipeline
    pipelineInfo.basePipelineIndex = -1; // Optional : reference another pipeline that is about to be created by index

    // i. final : create the graphics pipeline :
    // vkCreateGraphicsPipelines : can take many parameters to create many graphic pipelines in a single call
    // the second parameter : specify the optional VkPipelineCache to be used. cache can be used to store and reuse data relevent to pipeline creation accross multiple calls to this function and even between program execution if stored in file. GOAL : speed up the pipeline creation
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &
        pipelineInfo, nullptr, & graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    // cleanup
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

  };

  // k. Create frame buffers : we have set a render pass to expect a single framebuffer with same format as the swap chain image, now we need to create a framebuffer

  // the attachments specified during render pass creation are bound by wrapping them unto VkFrameBuffer object
  // framebuffer object references all of the VkImageView object that represent the attachment (in our case the color attachment ). Which image we have to use for the attachment depends on which image the swap chain returns when we retrieve one for presentation. This means that we have to create a framebuffer for all the images in the swap chain and use the one that corresponds to the retrieved image at drawing time.
  void createFramebuffers() {

    // resize the vector holding all the frambuffers to be able to hold as many frame buffer as there is images in the swap chain
    swapChainFramebuffers.resize(swapChainImageViews.size());

    // iterate through the image views in the swap chain and create a frame buffer for each one.
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      VkImageView attachments[] = {
        swapChainImageViews[i]
      };

      // creating the frame buffer
      VkFramebufferCreateInfo framebufferInfo {};
      framebufferInfo.sType =
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = renderPass; // specify the render pass the frame buffer will be compatible with . you can use frambuffer with renderpass that is compatible (same number and type of attachments )

      // specify the VkImageView objects that should be bound to the respective attachment descriptions in the render pass pAttachement array.
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;

      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1; // the number of layers in image arrays (our swap chain is single images )

      if (vkCreateFramebuffer(device, & framebufferInfo, nullptr, &
          swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  };

    
    // l. before creating command buffers , we need to create command pools. these pools manages the memory that is used to store the buffers, then command buffers are allocated from them .
       void createCommandPool(){
           
           
           QueueFamilyIndices queueFamilyIndices =
           findQueueFamilies(physicalDevice);
           
           VkCommandPoolCreateInfo poolInfo{};
           poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
           // creation of command pool takes only 2 parameters:
           
           // parameter 1 : 2 possible flags : either allow command buffer to be recorded individually (otherwise they are reset together), or hint that the command buffers are recorded with new commands very often which may change memory allocation behaviour.
           // in our case, we will be recording command buffer every frame, we need to be able to reset and record over it (individually)
           poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
           
           
           // parameter 2 : command buffers are executed by submitting them on one of the device queues (similar to the presentation queue and graphics queue we retrieved.)
           // Each command pool can only allocate command buffers that are submitted on a signal type of queue .
           // we want to record draw commands , we will choose the graphics queue.
           poolInfo.queueFamilyIndex =
           queueFamilyIndices.graphicsFamily.value();
           
         
           if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
           VK_SUCCESS) {
           throw std::runtime_error("failed to create command pool!");
           }
           
           
           // Each command pool can only allocate command buffers that are submitted on a signal type of queue .
           // we want to transfer data from staging buffer to vertex buffer using buffer copy
           // buffer copy require transfer queue (optional)
           // we need to create another command pool to record buffer copy using the transfer queue.
          /* VkCommandPoolCreateInfo poolTransferInfo{};
           poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
           poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
           poolInfo.queueFamilyIndex =
           queueFamilyIndices.transferFamily.value();
           if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
           VK_SUCCESS) {
           throw std::runtime_error("failed to create command pool with transfer queue!");
           }
           */
       };

    //m.   Add a texture steps :
    //1. Create image object backed with device memory : similar to creation of vertex buffer :
    // - create staging ressource (can be a staging image or a staging buffer) and fill it with pixel data
    // create image to copy to : images are like buffers, thye have memory attached to it :
    // - allocate the memory : query the memory requirement and allocate it
    // - bind the memory allocated with the image
    // - copy from the staging ressource (sometimes faster to copy from staging  buffer to image) to the final image used for rendering.
    //2. Fill this image with pixels from image file that we load
    // there is many image librarries to load images : ex: stb__image
    // 3. create an image sampler
    //4. add combined image sampler descriptor to sample colors from the texture

    void createTextureImage() {

      // 1.load an image and upload it into vulkan image object :
      int texWidth, texHeight, texChannels;
      // the stbi function takes the file path and the number of channels to load
      // STBI_rgb_alpha : forces the image to be loaded with alpha channe; even if it does not have one
      // it outputs the width, height, number of channels
      // it returns a pointer to the first element of array of pixel values
      // the array layout : row by row with 4 bytes per pixels (in case alpha is included )
      // the size of the array is : number of pixels * 4 bytes = width * height * 4 bytes.
      stbi_uc * pixels = stbi_load("/Users/sanajallouli/Downloads/sanamine.jpg", & texWidth, &
        texHeight, & texChannels, STBI_rgb_alpha);
      VkDeviceSize imageSize = texWidth * texHeight * 4;

      if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
      }

      //2. create staging buffer, allocate memeory to it according to the requirements and associate the memeory to the buffer .
      createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
        stagingBufferMemory);

      //3. copy the pixels of the loaded image into the mapped mempry of the staging buffer ;
      //3.a map the memory of the staging buffer
      void * data;
      vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, & data);
      //3.b copy the pixels into the mapped memory
      memcpy(data, pixels, static_cast < size_t > (imageSize));
      // unmap the memory
      vkUnmapMemory(device, stagingBufferMemory);

      // 4. clean the original pixel array
      stbi_image_free(pixels);

      //5. Create an image, whre we will copy the data from the staging buffer to. the shader will then access this image.
      // the shader could have accessed a buffer , but uit wil be faster to access image , as wit s able to use 2d coord
      // creating image requires filling up VkImageInfo:
      createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage,
        textureImageMemory);

      // 6. copy the data from the staging buffer to the image: this is done in the gpu (so requires use of command buffer). This requires first the data to be in the right layout :
      // 6.a transition the image into the right layout
      // the layout transition needs to submit commands to the gpu , handled by the function :
      transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED, // we do not care about the initial layout of the image
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL); // preapre for the shader access

      // 7. copy from the staging buffer with the pixels into the image
      // this helper function handles the recording of the command to copy the pixels in the GPU
      copyBufferToImage(stagingBuffer, textureImage,
        static_cast < uint32_t > (texWidth),
        static_cast < uint32_t > (texHeight));

      //8. cleanup the staging buffer and iits memeory
      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);

    }
    
    
    //n.
  void createTextureImageView() {
    // same code as createImageViews, just change the format and the image
    textureImageView = createImageView(textureImage,
      VK_FORMAT_R8G8B8A8_SRGB);
  }

    
    // Samplers :
    /*
     It i spossible for shaders to read direcly the texels (picels in the image) from images , but it is not very common when the image is used as texture.
     Textures are generally accessed through samplers. Samplers apply filtering and transformations to compute the final color.
     
     - Samplers can apply useful filters :
     The filters applied by the samplers are helpful to deal with problems like oversampling.
     ex:
        - Oversampling :
     Imagine a texture that is mapped to geometry with more fragments than texels.  If you simply took the closest texel for texture corrdinate in each fragment, you would get a result very pixelised. Imagine you divide the texture into normaiized coordinates and the the geometry mapped into normalized coordinates, you will map the uv and take the color of the dominenet texel where the coordinate fell in.
        However, if combine the 4 closest texels thgough linear interpolation, you would get a smoother result (blur).
        - Undersampling : a texture that is mapped to a geometry with less fragments than it has texels. This leads to artifacts when sampling high frequency patterns like checkerboard texture at sharp angles. this results into a blurry mess . The solution applied by the sampler is anisrtopic filtering.
     
     - Samplers can take care of transformations. It determines what happens when you try to read texels outside the image through addreseing mode : repeat, mirrored repeat. cleamp to edge . clamp to border.
     
     */

    VkSampler textureSampler; // hold the handle of the sampler object. note that the sampler is a distionct object that is not realted to the image and is not bound to any view or image.

    // o. set up sampler : shaders generally access textures trghough samplers. they can appply useful filter when reading the texture and take care of transformations
    void createTextureSampler() {
      // configure the sampler : specify the filters and transformations
      VkSamplerCreateInfo samplerInfo {};
      samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      // how to interpolate texels taht are magnified and mignified
      samplerInfo.magFilter = VK_FILTER_LINEAR; // magnifying concerns oversampling problems
      samplerInfo.minFilter = VK_FILTER_LINEAR; // concerns undersamplifying

      // The addressing modes : specified by axis (U,v , w instead of x,y,z which is convention for texture space coordinate)
      // this is visible when you use textyure coordinates above 1 and below 0 for norianlized coordinates for exemple.
      samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; // repeat the texture when going beyond the image dimension
      samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

      // unless performance is a concern, there is no reason not to use it.
      // note that this is a device feature so we need to request it.
      // also verify it is availabe for your device
      samplerInfo.anisotropyEnable = VK_TRUE;

      // limit the amount of texel samples that can be used to calculate the final color . Lower value insures better performace and lower quality. The values we can use depend on the physical device propertues.
      VkPhysicalDeviceProperties properties {};
      vkGetPhysicalDeviceProperties(physicalDevice, & properties);
      samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy; // maximum quality

      // which color is returned when sampling beyond the image with clamp to border addressing mode. Note that you cannot specify an arbitrary color.
      samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

      // Speciify whoch coordinate system you want to use to address texels in an image.
      // If this field is set to true: you can simply use coordinates within the [0, texWidth) and [0, texHeight) range , the the texwhodh and hight being the number of texels
      // vk_false: the texels are addressed using [0,1) range on all axes. This is the most common b/c then it possible to use textures of varying resolutions with same coordiinates.
      samplerInfo.unnormalizedCoordinates = VK_FALSE;

      // comparison function :
      // if vk_true : texels will first be compared to a value and the result of the comparison is used in filtering opeartions. Thios is used for percentage-closer filtering in shadow maps.
      samplerInfo.compareEnable = VK_FALSE;
      samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

      // fieds related to mipmapping , which is basicly another type of filtering to be applied.
      samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      samplerInfo.mipLodBias = 0.0f;
      samplerInfo.minLod = 0.0f;
      samplerInfo.maxLod = 0.0f;

      // create the sampler using the configuration struct
      if (vkCreateSampler(device, & samplerInfo, nullptr, &
          textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
      }
    }
    //p. createvertexbuffer with staging
    void createVertexBuffer(){
        
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
       
        // Staging buffer :
        // create a Staging buffer in CPU accessible memory to upload the data from the vertex array to
        // create a vertex buffer in device local memory
        // so that we we use buffer copy command to move the data from the staging buffer to the actual vertex buffer.
        
        // to be able to use buffer copycommand : we need a queue family that supports that
        // we need a queue that supports the VK_QUEUE_TRANSFER_BIT (any queue that has VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT already implicitly supports it = not required to have a specific queue family for this , we will do it anyway just to practive )
        
        // create a staging buffer with staging buffer memory for mapping and copying the vertex data
         VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        // ew specify the memory properties to indicate the transfer source flag so that we are able to transfer data from it to the vertex data (where we also indicated destination flag)
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // memory can be used as source in a memory transfer operation.  /  VK_BUFFER_USAGE_TRANSFER_DST_BIT : can be used as destination in a memory transfer operation
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);
      
        // Copy the vertex data into the staging buffer mapped memory
       
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0,
        &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
       vkUnmapMemory(device, stagingBufferMemory);
        
         
        // create vertex buffer : specify the mempry properties so that it is device local (generally means that we cannot use map memory on it), we can copy data from the staging buffer to it,
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | // memory can be used as destination in memory transfer operation
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
        vertexBufferMemory);
        
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        
    }

    // q.
    void createIndexBuffer(){
          VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
          
          // CRATE STAGING BUFFER :
          VkBuffer stagingBuffer;
         VkDeviceMemory stagingBufferMemory;
         // ew specify the memory properties to indicate the transfer source flag so that we are able to transfer data from it to the vertex data (where we also indicated destination flag)
         createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // memory can be used as source in a memory transfer operation.  /  VK_BUFFER_USAGE_TRANSFER_DST_BIT : can be used as destination in a memory transfer operation
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
         stagingBuffer,
         stagingBufferMemory);
       
          // Copy the indices data into the staging buffer mapped memory
         void* data;
         vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0,
         &data);
         memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
         
          createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | // memory can be used as destination in memory transfer operation
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
                       indexBufferMemory);
          
          // now this buffer will be used in a command buffer where we copy the data from the staging to the index buffer:
          // the method already contain the creation of the command buffer and registered the command ...etc.
          copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
          
          vkDestroyBuffer(device, stagingBuffer, nullptr);
          vkFreeMemory(device, stagingBufferMemory, nullptr);
      }

    /*
        Compute shaders  :
       Vulkan support compute shaders no matter th device (unlike old apis like OpenGL ). ALLow GPGPU (General purpose ocmputing on GPU), meaning allow general computing that used to be done on CPU is not possible on GPU. GPUs have become flexible and powerful to be able to take the general purpose tasks that used to be done in CPU . these tasks can now be done in GPU in real time.
        ex of tasks where thec compute capability of GPU can be used : image manipulation, visibility testing, post processing, advanced lighting calculation, amimation, physics.. and tasks that does not require graphics output : AI realted thigs , number crunching ...etc.  Headless computing.
        
        Advantage of using GPU instead of CPU for these computationally expensive tasks:
       -  offloading work from CPU
        - not requiring to move data form GPU to CPU (data can stay in GPU without having to wait for slow transfer)
        - take advantage of the parallelization of the GPU : better fit for higly parallel workflows : GPU have thousands of small compute units, CPU has few large compute units.
        
        
        Compute shader is not part of the graphics pipeline : we can use it anywhere where it fits. (not same as other shaders ex: fragment shader: always applied to the transformed output of the vertex shader. )
        
        Note that we can use descriptor sets in the compute.
        
        
        Exemple of use of compute shader :
        Particle system : thousands of particles that need to be updated almost every frame based on some equation.
        To render these particles we need : vertices passed as vertex buffer to the vertex shader and a way to update them based on an equation.
        CPU based barticle system : store the particle data in the main memory of the system and use the CPU to update them. Once updated, the updated vertices are transferred to gpu memory so that they can be displayed. (the most obvious way is to reacrate the verex buffer each time, or mapping the gpu memory so it can be written by the CPU = "RESIZABLE BAR" or "unified memory" on intergrated GPU", host local buffer ) : this is  costly beacuse all the buffer update methods requires a round trip to the cpu to update the particles. the bandwidth is limited to PCI-expres which is a fraction of the GPU bandwidth.

        GPU based particle system : does not require the roundtrip to the CPU.  The vertices are uploaded to the gpu at the start an dll updates are done in the GPU's memory using compute shaders.FASTER also because of the higher bandwidth between the gpu and it s local memory. with a dedicated compute queue, you can update particles in parallel to the rendering part of the graphics pipeline = ASYNC COMPUTE.
        
        
        Note that compute shader uses queue with property flag bit : VK_QUEUE_COMPUTE_BIT.
        YTou can wither use dedicated queue (without graphics bit), or queue with graphics bit and compute bit. If you use dedicated, it is a hint for ASYNCHRONOUS compute queue (update stuff in paralllel to the rendering part of the graphics pipeline )
        */

       /*
        
        The ability to arbitrarly READ and WRITE to buffers is done through 2 dedicated strorage types :
        - Shader storage buffer object (SSBO) : allows shader to readfrom and write to a buffer . using it is similar to using uniform buffer objects. the difference is that uou can alias other buffer types and they can be arbitrarily large.
               - use of SSBO in the particle system exemple : vertices are updated (written) by the compute shader and read from the vertex shader. this can be done using the same buffer b/c you can specify different usages for buffers and images. This buffer needs to be used as vertex buffer (to allow per vertex references ? ) ans as storage buffer for the compute shader to write to it. just specify 2 falgs in the ubuffer usage.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT; these flags tell the implementation that we want to use this buffer for 2 scenarios. We also add VK_BUFFER_USAGE_TRANSFER_DST_BIT to be able to transfer data from host to gpu b/c we want the shader storage to stay in gpu memory and we need to transfer data form the host to it.
        In glsl his is how ssbo are declared :
        struct Particle {
        vec2 position;vec2 velocity;vec4 color;
        };

        layout(std140, binding = 1) readonly buffer ParticleSSBOIn { // std140 : memory layout qualifiuer : determines how the memeber elements of the shader storage buffer are aligned in memory. required to map the buffers between host and gpu
        Particle particlesIn[ ];  // unbound number of particles (not having to specify the number os an advantage over uniform buffers )
        };

        layout(std140, binding = 2) buffer ParticleSSBOOut {
        Particle particlesOut[ ];};
        
        
       - Storage Images : allows you to read and write to an image. Ex : applying image effects to textures , post-processing or generating mipmaps. It is similar to creating SSBO : specify the 2 udages of the image at its creation imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT; // these two flags tells the implementation that we want to use this image for 2 different scenarios : smaple form the image in the fragment shader and store in it in the compute pass.
        In glsl, this is how the storage image is accessed :
        layout (binding = 0, rgba8) uniform readonly image2D inputImage;
         layout (binding = 1, rgba8) uniform writeonly image2D outputImage;
        */
       /*
        Work groups & Invocations : they define an abstract execution model for how compute workloads are processed by the compute hardware of the GPU in 3 dimensions x,y and z.
        
        - Work groups : how the compute workload are formed and processed by the compute hardware of the GPU. imagine work groups as work items the GPU has to work through. The work groups dimensions are set by the application at command bugger time using a DISPATCH command.
        Each group is a collection of INVOCATIONS that execute the same compute shader. The invocations can potentially run in parallel and their dimensions are set in the compute shader. Invocations within the same group have access to shared memory.
        The dimension of the work groups and the invocations is determined by the coommand byffer depend on how the input data is structured. ex: 1d array : only the x dimaension is specified for the work group and the invocation.
        ex: dispatching work group of [64, 1,1] (dimension of the work group) and compute shader local size (dimension of invocation) [32,32,1]: the compute shader will be invoker : 64 * 32* 32
        
        // the maximum cont for work groups and local sizes differs fro mimplementation to implementation/
        -
        
        */
       
       // to use compute shader :
       /*
        1. Select a queue family that supports COMPUTE : VK_QUEUE_COMPUTE_BIT when creating the physical device , store queue handle in compute queue when createing the logical device
        
        2. load compute shader : similar to loading vertex shader into the app : the difference is the type of shaders and the where to bind it in the pipeline :
        read the spv compute shader , specify its stryuctre VkPipelineShaderStageCreateInfo (type of shader , the method to run in it .. )
        
        
        3. create the storage buffer : create as many buffers as we have frames in flight, create the particles data and transfer it to these buffers using staging buffer.
        
        4. the compute stage needs to access the storage buffer : use descriptors :
               4.A : Specify it in the descriptor set layout similar to other descriptor , just specify : VK_SHADER_STAGE_COMPUTE_BIT and its type being ssbo. Not ethat here we specify the binding etc. so if you need to access 2 ssbo at same time, specify 2 distinct layout
               4.B : The descriptor pool is responsible to allocate the descriptor sets : include the new descriptor type in it.
               4.C : Create the descriptor sets using the descriptor pool : bind the descriptor to the memeory of the buffers and say how to update them
        
         5. compute shader is not part of the graphics piepline: create a dedeciated compute pipeline. iT HAS A LOT LESS STATES THAN GRAPHICS PIPELINE.
        
        6. write the compute shader
        
        7. Running the compute commands : tell the GPU to do some compute.
           Dispatch is for compute pipeline, what draw call is for graphics pipeline.
               - record the work and submit it in command buffer : submit the dispatch command : each frame redispatch (when the graphics queue is submitted after this, it will use the updated data after the shader has done the computations). this needs synchronization.
        
       8. Add synchroniaztion between vertex reading and compute shader writing : add fences and semaphores in the createSynchronizationObjects
        
        
        
        
        */

       std::vector < VkBuffer > shaderStorageBuffers;
       std::vector < VkDeviceMemory > shaderStorageBuffersMemory;
    
    //r.
       void createShaderStorageBuffers(){

           // Initialize particles
           std::default_random_engine rndEngine((unsigned)time(nullptr));
           std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
           
           // Initial particle positions on a circle
           std::vector<Particle> particles(PARTICLE_COUNT);
           for (auto& particle : particles) {
           float r = 0.25f * sqrt(rndDist(rndEngine));
           float theta = rndDist(rndEngine) * 2 *3.14159265358979323846;
            float x = r * cos(theta) * HEIGHT / WIDTH;
           float y = r * sin(theta);
         
           particle.position = glm::vec2(x, y);
           particle.velocity = glm::normalize(glm::vec2(x,y)) *0.00025f;
          particle.color = glm::vec4(rndDist(rndEngine),
           rndDist(rndEngine), rndDist(rndEngine), 1.0f);
            }
           shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
           shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
          
           // we will use a staging buffer to move the particle data from host (cpu) to shader storage buffer (gpu memory)
          
           VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
         
           VkBuffer stagingBuffer;
           VkDeviceMemory stagingBufferMemory;
           createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
           stagingBufferMemory);
       
           void* data;
           vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0,&data);
           // copy the data from the cpu to the staging buffer
           memcpy(data, particles.data(), (size_t)bufferSize);
           vkUnmapMemory(device, stagingBufferMemory);

           for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
               // create the storage buffer and its relative memory
             createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBuffers[i],
                           shaderStorageBuffersMemory[i]);
               // copy data from staging buffer to storage buffer : copy buffer : record commands and submit them
               copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);

           }
           
       }

    // s.

  void createUniformBuffers() {

    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

      // persistent mapping : the buffer stays mapped to the pointer where the data is later written in . this increases the performance by not having to to map the memory each time.
      vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, & uniformBuffersMapped[i]);
    }

  }

  // t. note that pool size not adequate may not be caght by the validation layer. the drivers may try to solve theis internally : sometimes the drivers let us gate away with allocation that exceeds the limit of our descriptor pool and other tme il will fail VK_ERROR_POOL_OUT_OF_MEMORY
  void createDescriptorPool() {
    // 1. define which descriptor types the descriptor sets are going to contain and how many of them using  VkDescriptorPoolSize
    std::array < VkDescriptorPoolSize, 3> poolSizes {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount =
      static_cast < uint32_t > (MAX_FRAMES_IN_FLIGHT); // allocate one of the descriptors for every frame.
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount =
      static_cast < uint32_t > (MAX_FRAMES_IN_FLIGHT);
      
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount =
      static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2; // 2 for each frame in fligh
      
      

    // reference the poolsize in the VkDescriptorPoolCreateInfo
    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast < uint32_t > (poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    // specify the maximum number of descriptor sets that may be allocated
    poolInfo.maxSets = static_cast < uint32_t > (MAX_FRAMES_IN_FLIGHT);
    // optional : you can set a flag to say if the individual descriptor sets can be freed or not.

    // create the descriptor pool, and store its handle in descriptorPool
    if (vkCreateDescriptorPool(device, & poolInfo, nullptr, &
        descriptorPool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  // u. allocate the descriptor sets themselves, cannot be done directly needs to use descriptor pools :
  //  note : no need to explicitly clean descriptor sets , they are autonatically freed when the descriptor pool is destroyed
  void createDescriptorSets() {

    // fill VkDescriptorSetAllocateInfo

    std::vector < VkDescriptorSetLayout > layouts(MAX_FRAMES_IN_FLIGHT,
      ComputeDescriptorSetLayout); // descriptorSetLayout is object holding the descriptors bindings combined (description of the actual type of buffer or images . their number , where they will be used etc... ) : in our case it holds the binding with the ubo info (binding 0, uniform buffer, one struct, to be used in vertex shader )
    VkDescriptorSetAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    // specify the descriptor pool to allocate from
    allocInfo.descriptorPool = descriptorPool;
    // specify the number of descriptor sets to allocate
    allocInfo.descriptorSetCount =
      static_cast < uint32_t > (MAX_FRAMES_IN_FLIGHT);
    // specify descriptor layout to base them on
    allocInfo.pSetLayouts = layouts.data(); // the descriptor layout (array of descriptor bindings)

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT); // we will create one descriptor set for each frame on flight , all with the same layout
    //vkAllocateDescriptorSets will allocate descriptor sets, each with uniform buffer descriptor
    if (vkAllocateDescriptorSets(device, & allocInfo, descriptorSets.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    // now the descriptor sets have been allocated, but the descriptors within still need to be configured.
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // descriptors that refer to buffers (uniform buffer descriptors) are configured using VkDescriptorBufferInfo
      // bind the actual buffer to the descriptor set :
      VkDescriptorBufferInfo bufferInfo {};
      bufferInfo.buffer = uniformBuffers[i]; // specify the buffer : we have as many uniform buffers as we have frame in flight

      // region within the buffer that contains the data for the descriptor : offset +size define the region
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject); // b/c we are overwriting the whole buffer, it s also possible to use VK_WHOLE_SIZE

      // bind the actual image and sampler ressources to the descriptor set :
      VkDescriptorImageInfo imageInfo {};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = textureImageView;
      imageInfo.sampler = textureSampler;
        
        VkDescriptorBufferInfo storageBufferInfoLastFrame{};
        storageBufferInfoLastFrame.buffer = shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT];
        storageBufferInfoLastFrame.offset = 0;
        storageBufferInfoLastFrame.range = sizeof(Particle) * PARTICLE_COUNT;

        
        VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
        storageBufferInfoCurrentFrame.buffer = shaderStorageBuffers[i];
        storageBufferInfoCurrentFrame.offset = 0;
         storageBufferInfoCurrentFrame.range = sizeof(Particle) *
        PARTICLE_COUNT;
        
      //UPDATE THE DESCRIPTOR :
      //The configuration of descriptors is updated using the vkUpdateDescriptorSets function: requires to fill VkWriteDescriptorSet
      std::array < VkWriteDescriptorSet, 3 > descriptorWrites {};
      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = descriptorSets[i]; // descrior sets object hold the descriptor sets for each frame on fligh. the descriptor set to update
      descriptorWrites[0].dstBinding = 0; // binding (we gave our uniform buffer binding index 0 )
      descriptorWrites[0].dstArrayElement = 0; // first index in the array that we want to update (we are not using an array so index is 0)

      // specify the type of descriptor (now uniform buffer) :
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      // how many array elements you want to update
      descriptorWrites[0].descriptorCount = 1;

      descriptorWrites[0].pBufferInfo = & bufferInfo; // used for descriptors that refer to buffer data
      //  descriptorWrite.pImageInfo = nullptr; // Optional : used for descriptor that refer to image data
      // descriptorWrite.pTexelBufferView = nullptr; // Optional : used for descriptors that refer to buffer views.

    
        descriptorWrites[1].sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i]; //
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;
        
        
        descriptorWrites[2].sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;
        
    
      // apply the update
      // it accepts 2 types of arrays : VkWriteDescriptorSet & VkCopyDescriptorSet (used to copy descriptors to each other)
        vkUpdateDescriptorSets(device, 3, descriptorWrites.data(), 0, nullptr);

    }

  }


    //v.
  void createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

    if (vkAllocateCommandBuffers(device, & allocInfo, commandBuffers.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }
  }

    std::vector < VkCommandBuffer > computeCommandBuffers;
    // w.
    void createComputeCommandBuffers() {
        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

      VkCommandBufferAllocateInfo allocInfo {};
      allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocInfo.commandPool = commandPool;
      allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocInfo.commandBufferCount = (uint32_t) computeCommandBuffers.size();

      if (vkAllocateCommandBuffers(device, & allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
      }
    }
    
  // command buffer recording : writes the commands we want to execute into command buffer that we already allocated
  // used in draw frame inside the main loop
  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

    // 1. begin recording commands : call vkBeginCommandBuffer and pass to it VkCommandBufferBeginInfo
    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional : specify how we are going to use the command buffer : command buffer will be recorded just after executing it once/ secondary command buffer that will be entirely within a single render pass / command buffer can be resubmitted while it is also already pending execution
    beginInfo.pInheritanceInfo = nullptr; // Optional : only relevant for secondary command buffers. specifies which state to inherit from the calling primary command buffer.

    // calling vkBeginCommandBuffer implicitly reset the command buffer it was already recorded once.
    // it is not possible to appends commands to the buffer at a later time.
    if (vkBeginCommandBuffer(commandBuffer, & beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    //2. start a render pass
    // drawing starts by beginning a render pass : vkCmdBeginRenderPass and pass to it render pass infos
    VkRenderPassBeginInfo renderPassInfo {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

    renderPassInfo.renderPass = renderPass; // specify the render pass object (wraps the attachments , here color buffer, to use and how to use it)

    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; // bind the frame buffer for the swap chain image where it is specified as a color attachment ( we have a frame buffer for each image in the swap chain, the frame buffer binds the image in the swap chain and the attachments specified in the render pass creation). pick the right frame buffer for the current swap chain image by using the imageIndex parameter passed to the function.

    // define the size of the render area (where the shader loads and stores will take place, the pixels outside this region will have undefined values, it should match the size of the attachment for best performance):
    renderPassInfo.renderArea.offset = {
      0,
      0
    };
    renderPassInfo.renderArea.extent = swapChainExtent;

    // the clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR which we used as load operation for the color attachment
      VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = & clearColor;

    // All the functions that record commands can be recognized by vkCmd prefix
    // the first param is always the command buffer
    // second param : specify the details of the render pass : renderPassInfo
    // third : how the drawing command within the render pass will be provided : either the render pass commands will be embedded in the primary command buffer itself and no secondary command buffer will be executed / or the render pass commands will be executed from secondary
    vkCmdBeginRenderPass(commandBuffer, & renderPassInfo,
      VK_SUBPASS_CONTENTS_INLINE);

    //3. Bind the graphics pipeline : tell Vulkan which operations to execute in the graphics pipeline and which attachments to use in the fragment shader
    // second param : specifies if the pipeline object is graphics or compute pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline); // graphics pipeline is the object holding the graphics pipeline

    // 4. specify if anything is set dynamic in the graphics pipeline : When defining the fixed function, we decided that the scissor and viewport for this pipeline are dynamic : now we need to specify them

    VkViewport viewport {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast < float > (swapChainExtent.width);
    viewport.height = static_cast < float > (swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, & viewport);

    VkRect2D scissor {};
    scissor.offset = {
      0,
      0
    };
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, & scissor);

    //5. Bind the vertex buffer during the rendering operations.
    VkBuffer vertexBuffers[] = {
      vertexBuffer
    };
    VkDeviceSize offsets[] = {
      0
    };
    // vkCmdBindVertexBuffers is used to bind vertex buffers to bindings
    // first two parameters : offset and number of bindings we are going to specify vertex buffers for.
    // last 2 parameters : specify the array of vertex buffers to bind and the byte offset to start reading vertex data from.
      
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffers[currentFrame], offsets);

    // 6.bind the index buffer :
    // 3rd parameter : byte offset
    // last parameter: type of index data
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    //7. Bind the right descriptor set for each frame to the descriptors in the shader
    //        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
    //                                pipelineLayout, // the layout that the descriptor is based on : bound to the descriptor set layout when setting up the graphics pipeline
    //                                0, // index of first descriptor set
    //                                1, // number of sets to bind
    //                                &descriptorSets[currentFrame], // the array of sets to bind
    //                                0, // array of offsets that are used for dynamic descriptors
    //                                nullptr);
    //
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, & descriptorSets[currentFrame], 0, nullptr);
    //8. issue the draw call :
    // 2nd param : vertex count (even though we technically did not specify vertex buffer we still have 3 vertices to draw )
    // 3rd param : instance count (used for instanced rendering, use 1 if not)
    // 4rth param : first vertex : offset into the vertex buffer (defines the lowest value of gl_VertexIndex)
    // 5th param: first instance : used as offset for instanced rendering (defines the lowest value of gl_InstanceIndex)

    // vkCmdDraw(commandBuffer, 3, 1, 0, 0); // draw command without vertex buffer
    // vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1,0, 0); // draw command with vertex buffer

    // draw command with index buffer : save memory
    // param 2 : number of indices (number of vertices that will be passed to the vertex shader , ex: for rectangle it is 6 vertices )
    // param 3:   number of instances (here not using instancing so we put 1 )
    // pram 4 : offset into the index buffer
    // pram 5 : offset to add to the indices in the index buffer
    // pram 6 / last param : offset for instantiating (not using instancing)
  //  vkCmdDrawIndexed(commandBuffer, static_cast < uint32_t > (indices.size()), 1, 0, 0, 0);
      vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);
    //9. End the render pass :
    vkCmdEndRenderPass(commandBuffer);

    //10. end recording the command buffer :
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }

  //R. create synchronization object
  /* Synchronization of execution on the GPU is explicit : order of operation is up to us to define using synchronization primitives which tell the driver the order we want things to run in
   Events that need synchronization, they are executed asynchronously, each one depend on previous one to be finished  :
   - acquire image from swap chain
   - execute commands that draw onto the acquired image
   - Present the image to the screen for presentation, returning it to the swap chain.
   
   
   2 places to apply synchronization :
   - swap-chain operations : happens in the GPU : use semaphores :
      - one semaphore to signal that the image has been acquired from the swapchain and is ready for rendering ,
      - one semaphore to signal that rendering has finished and presentation can happen.
   - waiting for the previous frame to finish : needs the cpu to wait until the previous frame finished : use fences because cpu (host )needs to wait so that it does nt draw more than one frame at a time = b/c we record the command buffer every frame, we cannot record the next frame's work to the command buffer until the current frame has finished to not override the current content of he command buffer while the gou is using it.:
      - one fence to make sure only one frame is rendering at a time.
   */

  /*
   Use of Semaphores : used to add order between the queue operations (the work we submit to a queue either in command buffer or from a function), ex of queue : graphics and presentation queue. Semaphore order the work inside a same queue and between different queues.
   
   2 kinds of semaphores in vulkan : Binary and Timeline (here only use binray)
   
   Semaphore is Signaled or Unsignaled (starts as unsignaled).
   
   How to use semaphores ?
   
   Provide the same semaphore S as a signal to one queue operation and as wait to the another queue operation :
   operation A will signal semaphore S when it finishes executing
   operation b will wait on semaphore S before it begins executing
   When operation B start executing, Semaphore S is reset back to being unsignaled so that it can be used again.
   
   VkCommandBuffer A, B = ... // record command buffers
   VkSemaphore S = ... // create a semaphore
   
   // enqueue A, signal S when done - starts executing immediately
    vkQueueSubmit(work: A, signal: S, wait: None)
   
    // enqueue B, wait on S to start
   vkQueueSubmit(work: B, signal: None, wait: S)
   
   The waiting happens in the GPU, the functio n calls return directly, the order is respected in the gpu
   **/

  /* Use of Fences : used for synchronization of execution on the CPU (known as the host) = When the host needs to know when the GPU has finished something we use a FENCE.
      = used to keep the cpu and GPU in sync with each other.
   
  
   Fence is either unsignaled or signaled .
   
   When we submit a work to execute, we attach a fence to that work. When the work is finished, the fence is signaled . The host wait for the fence to be signaled. This guarantees that the host only continues when the work is fjnished executing on the gpu side.
   
   ex of use : command to the gpu to take a screen shot and command to transfer the screen shot from gpu to cpu memory and save it to file. command buffer A executes the transfer , with fense F. submit command buffer A with fence F, tell the host to wait for F to signal. The host will wait until comand buffer A finishes executing, thus we are safe to let host save the file to disk.
   VkCommandBuffer A = ... // record command buffer with the transfer
   VkFence F = ... // create the fence
  
    // enqueue A, start work immediately, signal F when done
    vkQueueSubmit(work: A, fence: F)
  
    vkWaitForFence(F) // blocks execution until A has finished executing
  
    save_screenshot_to_disk() // can't run until the transfer has finished
   
   
   unlike the semaphore exemple, here the host is BLOCKED until the fence is signaled. but in general, it is preferable not to block the host unless necessary. we want to feed GPU and CPU with useful work not just waiting for fence.
   
   So, we prefer semaphore !
   
   Fences must be reset manually to put them back into unsignaled state. why ? because they are used to control the execution of the host, so he has to decide when to reset the fence. This is different from semaphore that is used to order the work on the gpu without the host being involved.
   */
  // x. create 2 semaphores and one fence
  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

    // Create the two semaphores : need to pass to it the filled VkSemaphoreCreateInfo
    VkSemaphoreCreateInfo semaphoreInfo {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    // creating fence : need to pass to it filled structure VkFenceCreateInfo
    VkFenceCreateInfo fenceInfo {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // create the fence on the signaled state , the first state of the fence is signaed.
    // wothout this , there would be a problem, because otherwise, we enter draw frame for the first time, we have to wait for the previous frame to finish, while there is no previous frame, and we would wait forever. like this, the first time we enter, it s already signaled, we can proceed.

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (vkCreateSemaphore(device, & semaphoreInfo, nullptr, &
          imageAvailableSemaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(device, & semaphoreInfo, nullptr, &
          renderFinishedSemaphores[i]) != VK_SUCCESS ||
        vkCreateFence(device, & fenceInfo, nullptr, &
          inFlightFences[i]) != VK_SUCCESS) {

        throw std::runtime_error("failed to create synchronization objects for a frame!");
      }
          // create the semaphore and fence for the compute shader
          if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
                       vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS) {
                       throw std::runtime_error("failed to create compute synchronization objects for a frame!");
                   }

      
    }
  }
   
   
    VkPipelineLayout computePipelineLayout;
       
       
       VkPipeline computePipeline;
       
    // j.
       void createComputePipeline(){
        
           //1. load the shaders :
           auto computeShaderCode = readFile("/Users/sanajallouli/Downloads/CODE PROJECTS SANA/VulkanProject/VulkanProject/shader4.comp.spv");
           
           //2. create shader module
           VkShaderModule computeShaderModule =
             createShaderModule(computeShaderCode);
           
           // 3. to use the sahders we will need to assign them to a specific pipeline stage
           
           //  specify the structure for compute shader
           VkPipelineShaderStageCreateInfo computeShaderStageInfo {};
           computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
           // tell vulkan in which pipeline stage the shader is going to be used. There is an enum for each programmable stage of the pipeline (ex: tessellation, vertex, fragment shaders)
           computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;

           // specify the shader module containing the code
           computeShaderStageInfo.module = computeShaderModule;
           // specify the function to invoke = entry point to the shader
           // this allows us to combine shaders into one shader module and specify different entry point to differentiate their behaviour
           computeShaderStageInfo.pName = "main";
           // specify values for shader constants can be specified here as well : pSpecializationInfo

           
           //4. create the pipeline layout
           
           VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
           pipelineLayoutInfo.sType =
           VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
           pipelineLayoutInfo.setLayoutCount = 1;
           // pipeline layout : the uniform and push values referenced by the shaders
           pipelineLayoutInfo.pSetLayouts = &ComputeDescriptorSetLayout;
       
           if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
           &computePipelineLayout) != VK_SUCCESS) {
           throw std::runtime_error("failed to create compute pipeline layout!");
           
           }
           
           // 5. create the compute pipeline : fill VkComputePipelineCreateInfo
           VkComputePipelineCreateInfo pipelineInfo{};
       
           pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
           pipelineInfo.layout = computePipelineLayout; // spcify the layout we just created
           pipelineInfo.stage = computeShaderStageInfo; // specify the shader module just created.
           
           if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
           &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
           throw std::runtime_error("failed to create compute pipeline!");
           }
           
           
         }
    
    // MAIN LOOP FUNCTION :
    
    //operations in draw frame are asynchronous, so when we exit the main loop, rendering and oresenting may still be going on.
    void drawFrameWithComputePass() {
      /* Rendering a frame in Vulkan :
       -1. Wait for previous frame to finish : host waits for previous frame to finish
       -2. acquire an image from the swap chain
       -3.  record a command buffer which draws the scene onto that image
       -4. submit the recorded command buffer
       -5. present the swap chain image
       */
      // update the uniform buffer :

      //1. Wait for
        vkWaitForFences(device, 1, &computeInFlightFences[currentFrame],
        VK_TRUE, UINT64_MAX);
        updateUniformBuffer(currentFrame);
        
        
        vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
      // VK_TRUE = want to wait for all the fences to finish
      //UINT64_MAX = timeout parameter , here this disables the timeout
        vkResetCommandBuffer(computeCommandBuffers[currentFrame],
        /*VkCommandBufferResetFlagBits*/ 0);
 recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);
        
        VkSubmitInfo submitInfo {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores =
        &computeFinishedSemaphores[currentFrame];

        
        if (vkQueueSubmit(computeQueue, 1, &submitInfo,
        computeInFlightFences[currentFrame]) != VK_SUCCESS) {
         throw std::runtime_error("failed to submit compute command buffer!");};
                                    
      //2. acquire an image form the swap chain :
      uint32_t imageIndex;
      // first two parameters : logical device and swap chain from which we want to acquire an image
      // 3rd parameter: time-out in nano sec
      //4rth, 5th params : image available semaphore/ null handler:  two parameters : specify the synchronization objects that are to be signaled when the presentation engine is finished using the image. this is when we can start drawing on it.
      // last param : imageIndex : variable to output the index of the swap chain image that has become available. it is the index of the swap chain image that has become available. this is then  used to pick up the corresponding frame buffer
    
        
        // graphics submission :
        
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
        UINT64_MAX);
        
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, & imageIndex);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);
   
         vkResetCommandBuffer(commandBuffers[currentFrame],
        /*VkCommandBufferResetFlagBits*/ 0);
         recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        
      updateUniformBuffer(currentFrame);

      //3. record the command buffer :
      vkResetCommandBuffer(commandBuffers[currentFrame], 0);
      recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

      //4. Submit the command buffer :
      // queue submission and synchronization is done by filling VkSubmitInfo
        VkSemaphore waitSemaphores[] = {
        computeFinishedSemaphores[currentFrame],
        imageAvailableSemaphores[currentFrame] };
         VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
         submitInfo = {};
         submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      // specify which semaphores to wait on before execution begins and in which stages of the pipeline to wait.
    
      // stage of the pipeline to wait in : wait with writing color to the image, until the image is available : the stage in the pipeline that writes to the color attachment. Each entry in the waitStages corresponds to the semaphore of same index in the waitSemaphores
     
        submitInfo.waitSemaphoreCount = 2;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
         submitInfo.commandBufferCount = 1;
         submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
         submitInfo.pSignalSemaphores =
        &renderFinishedSemaphores[currentFrame];

    
      if (vkQueueSubmit(graphicsQueue, 1, & submitInfo, inFlightFences[currentFrame]) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
      }

      //5. Presenting : submit the result back to the swap chain to have it eventually show up on the screen
      VkPresentInfoKHR presentInfo {};
      presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

      // which semaphore to wait on before presenting :
      // we want to present when the command buffer finishes executing
      // wait for the semaphores that are signaled at the end of command buffer execution
      presentInfo.waitSemaphoreCount = 1;
      presentInfo.pWaitSemaphores =  &renderFinishedSemaphores[currentFrame]; //(here signal semaphore signals that the rendering happened and we are ready to present)

      // specify the swap chain to present images to
      VkSwapchainKHR swapChains[] = {
        swapChain
      };
      presentInfo.swapchainCount = 1;
      presentInfo.pSwapchains = swapChains;
      presentInfo.pImageIndices = & imageIndex; // the index of the image for each swap chain

      presentInfo.pResults = nullptr; // Optional : specify an array VkResult values to check for every individual swap chain if presentation is succesful , not necessary when you have only one swap chain.

      // submit the request to present an image to the swap chain
      vkQueuePresentKHR(presentQueue, & presentInfo);

      // advance the frame :
      currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // use modulo to ensure that the frame index loops around after every Max_frame_InFLight enqued frames.
    }

      
    

  // variables
  GLFWwindow * window; // store reference to the window
  VkInstance instance; // handle to the instance
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // object to store the graphics card selected. this is implicitly destroyed when the instance is destroyed
  VkDevice device;
  VkPhysicalDeviceFeatures deviceFeatures {}; // no specific features for now
  VkQueue graphicsQueue; // store the queue handles
  VkQueue transferQueue; // store queue handle for transfer queue.
  VkSurfaceKHR surface;
  VkQueue presentQueue; // handle to the presentation queue
    VkQueue computeQueue; // handle to the presentation queue
  VkSwapchainKHR swapChain; // store the swapchain object
  std::vector < VkImage > swapChainImages; // contains the handles of the vkImages in the swap chain. The images has been cerated by the implementation of the swap chain
  VkFormat swapChainImageFormat; // the image format we specified when creating the swap chain (color channels, types, whether srbg supported )
  VkExtent2D swapChainExtent; // the extent we specified when creating the swap chain (the resolution of the swap chain images)
  std::vector < VkImageView > swapChainImageViews; // store the image views
  VkPipelineLayout pipelineLayout; // used to hold the uniforms ?
  VkRenderPass renderPass; // the render pass : the object that wraps all the infos about the color/ depth buffers and they samples.. etc.

  VkDescriptorSetLayout descriptorSetLayout; // all the descriptor bindings are combined into a single object.

  VkPipeline graphicsPipeline; // hold the pipeline object
  std::vector < VkFramebuffer > swapChainFramebuffers; // hold the different framebuffers: one framebuffer for image in the swap chain, ans use the one that corresponds to the retrieved image at drawing time.
  VkCommandPool commandPool; // store command pool object, the command pool manages the memory that is used to store the buffers, so that command buffer can be allocated from them.

  VkCommandBuffer commandBuffer; // object  where we record all the operations before we submit them to queue to be executed.

  // need 3 synchronization objects :
  VkSemaphore imageAvailableSemaphore; // semaphore to signal that an image has been acquired by the swap chain (presentation has finished )and is ready for the rendering
  VkSemaphore renderFinishedSemaphore; // signal that rendering has finished and presentation can happen
  VkFence inFlightFence; // fence that make sure that only one rendering is happening at a time

    std::vector<VkFence> computeInFlightFences;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    
    
  // Frame in flight : each frame should have its own ressources:: command buffer, semaphore, fence.
  std::vector < VkCommandBuffer > commandBuffers;
  std::vector < VkSemaphore > imageAvailableSemaphores;
  std::vector < VkSemaphore > renderFinishedSemaphores;
  std::vector < VkFence > inFlightFences;
  std::vector <
  const char * > deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };

  // we should store both of them in same VkBuffer for better performance and allocate them from single memory allocation! In that case it s more cache friendly as it is closer together
  // Aliasing : reusing the same chunk of memory for multiple resources if they are not using during the same rendering operations, provided that their data is refreshed.
  VkBuffer vertexBuffer; // hold the buffer handle
  VkBuffer indexBuffer;
  VkDeviceMemory vertexBufferMemory; // hold the handle to the memory , this memory is to be associated with the corresponding buffer (here vertexbuffer)
  VkDeviceMemory indexBufferMemory;

  // we should have multiple buffers , because multiple frames may be in flight at the same time and we don t want to update the buffer in preparation of the next frame while a previous one is still reading from it.
  // We should have as many uniform buffers as we have frame in flight, and write to the uniform buffer that is not currently being read from by the gpu.
  std::vector < VkBuffer > uniformBuffers;
  std::vector < VkDeviceMemory > uniformBuffersMemory;
  std::vector < void * > uniformBuffersMapped;

  VkDescriptorPool descriptorPool; // handle to the descriptor pool object
  std::vector < VkDescriptorSet > descriptorSets; // hold the descriptor set for each frame on flight

  const std::vector <
    const char * > validationLayers = {
      "VK_LAYER_KHRONOS_validation"
    };
  #ifdef NDEBUG // c++ macro  == not debug
  const bool enableValidationLayers = true;
  #else
  const bool enableValidationLayers = true;
  #endif

};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
