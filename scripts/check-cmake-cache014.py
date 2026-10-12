from pathlib import Path
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014/build')
for p in [r/'CMakeFiles/3.31.10/CMakeCXXCompiler.cmake',r/'CMakeFiles/3.31.10/CMakeCCompiler.cmake']:
 print('FILE',p)
 if p.exists():
  for line in p.read_text().splitlines():
   if any(k in line for k in ['COMPILER_ID','COMPILE_FEATURES','COMPILER_WORKS','ABI_COMPILED','COMPILER_FORCED','COMPILER_VERSION']): print(line)
