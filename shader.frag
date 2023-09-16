 #version 450

 // the fragment shader is invoked on each fragment to produce a color and depth for the framebuffer (bind image ref to render passes )

// there is no built in output variable, you have to specify the output variable for each framebuffer 
//  the layout(location =0) modifier specifies the index of the framebuffer 
// the color is outputted to the outcolor variable that is linked to the first and only framebuffer at index 0 .


layout(location = 0) out vec4 outColor;

// gets input from the vertex shader 
// not necessarily use the same name as the vertex shader
// the two will be linked together using the indexes specified by the locations directive  
layout(location = 0) in vec3 fragColor;
// just like per vertex color, the texture coordinates are smoothly interpolated across the area of the square by the rasterizer : visualize this by outputting the texture coordinates as colors.


 
void main() { // called for every fragment 
    //outColor = vec4(fragTexCoord, 0.0, 1.0);  // the values will be automatically interpolated for the fragments between the three vertices
    
    outColor =  vec4(fragColor , 1.0f); // built in texture function that samples textures : takes sampler and coordinate . sampler automatically takes care of filtering and transformations in the background.
}
