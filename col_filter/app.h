/* Copyright 2018 Stanford, UT Austin, LANL
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _COLFILTER_APP_H_
#define _COLFILTER_APP_H_
#include <cstdint>

//#define VERTEX_DEGREE
#define EDGE_WEIGHT

typedef uint32_t V_ID;
typedef uint64_t E_ID;
typedef int WeightType;

const float LAMBDA = 0.001;
const float GAMMA = 0.00000035;
const int K = 20;

struct NodeStruct {
  E_ID index;
  V_ID degree;
};

struct EdgeStruct {
  V_ID src, dst;
  WeightType weight;
};

struct Vertex {
  float v[K];
};

#endif
