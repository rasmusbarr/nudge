# nudge

Nudge is a small data-oriented and SIMD-optimized 3D rigid body physics library.

For more information, see: http://rasmusbarr.github.io/blog/dod-physics.html

## FAQ

### The sample application is crashing. Why?

Most likely, your CPU doesn't support AVX2 and/or FMA. The project files are set to compile with AVX2 and FMA support and you need to disable it in build settings.

**Xcode:** Set "Enable Additional Vector Extensions" to your supported level. Remove -mfma and -mno-fma4 from "Other C Flags".

**Visual Studio:** Set "Enable Enhanced Instruction Set" under code generation to your supported level. Remove \_\_FMA\_\_ from the preprocessor definitions.
