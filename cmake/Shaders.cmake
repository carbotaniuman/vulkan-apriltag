find_program(SLANGC slangc REQUIRED)

set(APRILTAG_SLANG_FLAGS
  -target spirv
  -profile spirv_1_6
  -capability spvVulkanMemoryModelKHR
  -fvk-use-scalar-layout
  -fvk-use-entrypoint-name
  -warnings-as-errors all
)

set(APRILTAG_EMBED_SPIRV ${CMAKE_CURRENT_LIST_DIR}/EmbedSpirv.cmake)

function(apriltag_add_shaders target)
  cmake_parse_arguments(ARG "" "" "SHADERS;INCLUDE_DIRS" ${ARGN})

  set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/shaders)
  file(MAKE_DIRECTORY ${out_dir})

  set(include_flags)
  foreach(dir IN LISTS ARG_INCLUDE_DIRS)
    get_filename_component(dir ${dir} ABSOLUTE)
    list(APPEND include_flags -I${dir})
  endforeach()

  set(headers)
  foreach(shader IN LISTS ARG_SHADERS)
    get_filename_component(src ${shader} ABSOLUTE)
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${src})
    string(MAKE_C_IDENTIFIER ${rel_path} var_name)
    set(header ${out_dir}/${rel_path}.h)
    set(spv ${out_dir}/${rel_path}.spv)
    get_filename_component(header_dir ${header} DIRECTORY)
    file(MAKE_DIRECTORY ${header_dir})

    # The SPIR-V comes first: slangc's depfile names it as the target, and
    # CMake reads the depfile for the first output.
    add_custom_command(
      OUTPUT ${spv} ${header}
      COMMAND ${SLANGC}
              ${APRILTAG_SLANG_FLAGS}
              $<IF:$<CONFIG:Debug>,-g,-O2>
              ${include_flags}
              -depfile ${header}.d
              -o ${spv}
              ${src}
      COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${header}
              -DVAR=${var_name} -P ${APRILTAG_EMBED_SPIRV}
      DEPENDS ${src} ${APRILTAG_EMBED_SPIRV}
      DEPFILE ${header}.d
      COMMENT "Compiling shader ${rel_path}"
      COMMAND_EXPAND_LISTS
      VERBATIM
    )
    list(APPEND headers ${header})
  endforeach()

  target_sources(${target} PRIVATE ${headers})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()
