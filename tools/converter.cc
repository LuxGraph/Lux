#include <map> 
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
typedef uint32_t V_IDX;
typedef uint64_t E_IDX;

void parse_input_args(char** argv, int argc, V_IDX &nv, E_IDX &ne,
                      std::string &input, std::string &output)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-nv"))
    {
      nv = atoll(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-ne"))
    {
      ne = atoll(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-input"))
    {
      input = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-output"))
    {
      output = std::string(argv[++i]);
      continue;
    }
  }
}

#define MAX_SIZE 65536

void add_to_buffer(V_IDX* buf, V_IDX src, size_t &buf_size, FILE* output)
{
  buf[buf_size] = src;
  buf_size++;
  if (buf_size == MAX_SIZE)
  {
    fwrite(buf, sizeof(V_IDX), MAX_SIZE, output);
    buf_size = 0;
  }
}

void flush_buffer(V_IDX* buf, size_t buf_size, FILE* output)
{
  if (buf_size > 0)
  {
    fwrite(buf, sizeof(V_IDX), buf_size, output);
  }
}

struct EdgeStruct
{
  V_IDX src, dst;
};

static bool compareLess(const EdgeStruct& a, const EdgeStruct& b)
{
  return a.dst < b.dst;
}

int main(int argc, char **argv)
{
  //assert(sizeof(unsigned) == 4);
  //assert(sizeof(size_t) == 8);
  E_IDX ne = 0;
  V_IDX nv = 0;
  std::string input = "", output = "";
  parse_input_args(argv, argc, nv, ne, input, output);
  printf("nv = %d ne = %lu input = %s output = %s\n", nv, ne, input.c_str(), output.c_str());
  FILE* fin = fopen(input.c_str(), "r");
  FILE* fout = fopen(output.c_str(), "wb");
  V_IDX* degrees = (V_IDX*) malloc(((size_t)nv) * sizeof(V_IDX));
  memset(degrees, 0, sizeof(V_IDX) * (size_t)nv);
  std::vector<EdgeStruct> edges(ne);
  for (E_IDX e = 0; e < ne; e++)
  {
    V_IDX src, dst;
    fscanf(fin, "%i %i", &src, &dst);
    assert(src < nv);
    assert(dst < nv);
    degrees[src]++;
    edges[e].src = src;
    edges[e].dst = dst;
    if (e % 100000 == 0) printf("%lu\n", e);
  }

  std::sort(edges.begin(), edges.end(), compareLess);
  printf("CP#1\n");
  E_IDX* row_ptrs = (E_IDX*) malloc(((size_t)nv) * sizeof(E_IDX));
  E_IDX cnt = 0;
  for (V_IDX v = 0; v < nv; v++)
  {
    while (cnt < ne && edges[cnt].dst == v) cnt++;
    row_ptrs[v] = cnt;
  }
  printf("CP#2\n");
  fwrite(&nv, sizeof(V_IDX), 1, fout);
  fwrite(&ne, sizeof(E_IDX), 1, fout);
  fwrite(row_ptrs, sizeof(E_IDX), (size_t)nv, fout);
  printf("CP#3\n");
  V_IDX* col_idx = (V_IDX*) malloc(MAX_SIZE * sizeof(V_IDX));
  size_t buf_size = 0;
  E_IDX start_col_idx = 0, end_col_idx = 0;
  for (V_IDX v = 0; v < nv; v++)
  {
    if (v % 100000 == 0) printf("v = %lu\n", v);
    start_col_idx = end_col_idx;
    end_col_idx = row_ptrs[v];
    for (E_IDX e = start_col_idx; e < end_col_idx; e++)
      add_to_buffer(col_idx, edges[e].src, buf_size, fout);
  }
  flush_buffer(col_idx, buf_size, fout);
  fwrite(degrees, sizeof(V_IDX), (size_t)nv, fout);
  fclose(fin);
  fclose(fout);
  free(row_ptrs);
  free(degrees);
  free(col_idx);
}
