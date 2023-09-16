#version 450

// uniform buffer object to access access the VkBuffer containing the resource we want
// the binding directive (binding =0) is going to be referenced in the descriptor layout (when we create the pipeline, we specify the descriptor layout to describe the type of resources the shader will accept)
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;


// pass arbitrary attributes to the vertex shader :
// these two variables are vertex attributes. they are specified per vertex in the vertex buffer.
// the layout location  assign indices to the inputs that we can later use to reference them.
// the location is specified when binding the vertex buffer in the creation of the graphics pipeline.
// NOTE THAT : Attention : some types takes multiples slots, meaning that the index after 0 is not 1 but 2 for exemple.
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;


// output for color from vertex to fragment shader
// this needs to have a matching input in the fragment shader
// not necessarly use the same name in both shaders 
// the two will be linked together using the indexes specified by the locations directive 
layout(location = 0) out vec3 fragColor;


 void main() {
 // invoked for every vertex
 // gl_VertexIndex contains the index of the current vertex. this is known because we specified the format of the data to be passed to the vertex shader 
 // this is usually an index into the vertex array (for now our hardcoded vertex array )
// this produces a position in the clip coordinate : dummy z and w
// gl_position is the built in output var, the triangle that is formed by the position from this vertex shader fills an area of the screen with fragments

 gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0); // the last parameter is the perspective division : make the obj closer
     

// pass the per- vertex color to the fragment shader
    // inColor is already per-vertex.
    // fragColor = colors[gl_VertexIndex];
fragColor = inColor;

 }

