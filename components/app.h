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
#ifndef _CC_APP_H_
#define _CC_APP_H_

#include <cstdint>
#define SPARSE_THRESHOLD 16
#define SLIDING_WINDOW 4 // Pre-launch more tasks to hide runtime overhead
typedef uint32_t V_ID;
typedef uint64_t E_ID;

struct NodeStruct {
  E_ID index;
};

typedef uint32_t EdgeStruct;

//struct EdgeStruct {
//  V_ID dst;
//};

struct EdgeStruct2 {
  V_ID src, dst;
};

typedef V_ID Vertex;
#endif
