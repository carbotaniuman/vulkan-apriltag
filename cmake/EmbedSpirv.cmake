file(READ ${INPUT} hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR rem "${hex_len} % 8")
if(hex_len EQUAL 0 OR NOT rem EQUAL 0)
  message(FATAL_ERROR "${INPUT}: not a whole number of 32-bit words")
endif()

string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1," words "${hex}")
set(word "0x........,")
string(REGEX REPLACE "(${word}${word}${word}${word}${word}${word}${word}${word})"
       "\\1\n\t" words "${words}")

file(WRITE ${OUTPUT}
  "#pragma once\nconst uint32_t ${VAR}[] = {\n\t${words}\n};\n")
