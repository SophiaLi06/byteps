#include <cstring>
#include <stdio.h>
#include <cmath>
#include <iostream>

#include "../compressor_registry.h"
#include "drive.h"

namespace byteps {
namespace common {
namespace compressor {
namespace{
CompressorRegistry::Register reg(
    "drive_compressor",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor>{

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                            [](unsigned x) {return x != 0;});
    
      /* Minghao */
      printf("Drive compressor size: %d, dtype: %d, seed: %d\n", size, dtype, seed);
      /////////////
      return std::unique_ptr<Compressor>(
          new DriveCompressor(size, dtype, seed));
    });
}

/*
 * Except for error-feedback and momentum, the underlying data of input
 * should never be changed. this is because input is still used in error
 * feedback if enabled.
 */

/* In-Place 1D Hadamard Rotate*/
template <typename index_t, typename scalar_t>
void DriveCompressor::HadamardRotate(index_t* dst, const scalar_t* src,
                                       size_t len) {
  // TODO: add an error msg?
  assert(len & (len-1) == 0);
  size_t h = 2;
  size_t hf;
  //TODO: can this process be paralleled in some way?
  while (h <= len){
    hf = h / 2;
    // view the gradient as a (len // h * h) tensor
    for (size_t i = 0; i < len / h; i++){
      for (size_t j = 0; j < hf; j++) {
        // update front half of each "row"
        dst[i * h + j] = src[i * h + j] + src[i * h + hf + j];
        // update back half of each "row"
        dst[i * h + hf + j] = dst[i * h + j] - 2 * src[i * h + hf + j];
      }
    }
    h *= 2;
  }
  float sqrt_d = std::sqrt(len);
  for (size_t i = 0; i < len; i++) dst[i] /= sqrt_d; 
  /* Minghao */
  printf("Hadamard Rotate Complete\n");
  std::cout << "dst: " << dst << "\n";
  std::cout << "src: " << src << "\n";
  /////////////
}

template <typename index_t, typename scalar_t>
tensor_t DriveCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                       size_t len) {
  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  // PACKING_SIZE values will be compressed into one chunk
  // (Each scalar value is represented by one bit, so 8 values in one byte
  // and sizeof(scalar_t) * 8 in one scalar_t)
  constexpr size_t PACKING_SIZE = sizeof(scalar_t) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE));
  // The total number of chunks
  const size_t chunk_num = (len + padding_len) / PACKING_SIZE;
  
  // In-Place 1D Hadamard Rotate
  // TODO: may need to modify len to make it into a power of 2?
  
  // NOTE: need to change the compression params dictionary to pass
  // parameter 'seed'
  if (_seed != 0){
    // if random number generator is not none
    for (size_t i = 0; i < len; i++){
      dst[i] = src[i] * (2 * _rng.Bernoulli(0.5) - 1);
    }
    HadamardRotate(dst, dst, len);
  }
  else{
    HadamardRotate(dst, src, len);
  }

  // Compute the scale
  float norm1 = 0.0, norm2 = 0.0;

  // TODO: can this be paralleled?
  for (size_t i = 0; i < chunk_num; i++){
    size_t start_index = i * PACKING_SIZE;
    index_t x = (dst[start_index] >= 0);
    norm1 += std::abs(dst[start_index]);
    norm2 += (dst[start_index] * dst[start_index]);

    for (size_t j = 1; i < PACKING_SIZE; i++){
      norm1 += std::abs(dst[start_index + j]);
      norm2 += (dst[start_index + j] * dst[start_index + j]);
      
      x <<= 1;
      // take the sign
      // ('1' for positve, '0' for negative)
      x |= (dst[start_index + j] >= 0);
      //dst[start_index + j] = 1.0 - (2 * (dst[start_index + j] < 0));
    }
    dst[i] = x;
  }
  
  // note norm2 is actually the square of the L2 norm
  float scale = norm2 / norm1;

  // append the scale to the end of the tensor
  float* scale_ptr = reinterpret_cast<float*>(&dst[chunk_num]);
  *scale_ptr = scale;

  /* Minghao */
  printf("Compress Complete\n");
  std::cout << "dst: " << dst << "\n";
  std::cout << "src: " << src << "\n";
  /////////////

  return {dst, chunk_num * sizeof(index_t) + sizeof(float)};
}

tensor_t DriveCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename scalar_t, typename index_t>
tensor_t DriveCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                        size_t compressed_size){
  static_assert(sizeof(scalar_t) == sizeof(index_t),
                "scalar_t should be the same size as index_t");
  constexpr size_t PACKING_SIZE = sizeof(scalar_t) * 8;
  const size_t chunk_num = (compressed_size - sizeof(float)) / sizeof(index_t);
  /* Minghao */
  std::cout << "dst: " << dst << "\n";
  std::cout << "src: " << src << "\n";
  printf("Decompress here0\n");
  printf("PACKING_SIZE: %d, chunk_num: %d\n", PACKING_SIZE, chunk_num);
  /////////////

  auto* scale_ptr = reinterpret_cast<const float*>(src + chunk_num);
  float scale = *scale_ptr;

  index_t* ptr = const_cast<index_t*>(src);
  if ((void*)dst == (void*)src) {
    ptr = reinterpret_cast<index_t*>(_buf.get());
    std::memcpy(ptr, src, compressed_size);
  }

  /* Minghao */
  printf("Decompress here1\n");
  /////////////

  // TODO: can this be paralleled?
  for (size_t i = chunk_num - 1; i >= 0; i--){
    index_t x = ptr[i];
    for (size_t j = PACKING_SIZE - 1; j >= 0; j--){
      // restore the sign
      // (1 for positive, -1 for negative)
      // TODO: not casting to float should be fine? as it will then be
      // divided by the float "sqrt_d" in HadamardRotate?
      int sign = ((x & 0x01) << 1) - 1;
      dst[i * PACKING_SIZE + j] = sign;
      x >>= 1;
    }
  }

  /* Minghao */
  printf("Decompress here2\n");
  /////////////

  // in-place Hadamard Transform (inverse)
  HadamardRotate(dst, dst, chunk_num * PACKING_SIZE);
  
  /* Minghao */
  printf("Decompress here3\n");
  /////////////

  // if random number generator is not none
  if (_seed != 0){
    // if random number generator is not none
    for (size_t i = 0; i < chunk_num * PACKING_SIZE; i++){
      dst[i] = dst[i] * (2 * _rng.Bernoulli(0.5) - 1);
    }
  }

  /* Minghao */
  printf("Decompress here4\n");
  /////////////

  // scale and return
  for (size_t i = 0; i < chunk_num * PACKING_SIZE; i++){
    dst[i] *= scale;
  }
  /* Minghao */
  printf("Decompress Complete\n");
  /////////////
  return {dst, _size};
}

tensor_t DriveCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif
  /* Minghao */
  //this->_decompress_call++;
  //this->_decompress_size = compressed.size;
  ///////////////
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps
