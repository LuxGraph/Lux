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
#ifndef _PAGERANK_H_
#define _PAGERANK_H_

#include <cstdint>
#define VERTEX_DEGREE

typedef uint32_t V_ID;
typedef uint64_t E_ID;

const float ALPHA = 0.15;

struct NodeStruct {
  E_ID index;
  V_ID degree;
};

struct EdgeStruct {
  V_ID src, dst;
};

typedef float Vertex;
#endif
