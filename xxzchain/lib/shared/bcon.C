// dllmain.cpp : DLL 애플리케이션의 진입점을 정의합니다.
#include "pch.h"
#include "stdio.h"
#include "iostream"
//#include "mkl"
#include "thread"
#include "vector"


using namespace std;

#ifndef BCON_H
#define BCON_H
#ifdef BCON_DLL
#ifdef BCON_EXPORTS

#else
#define BCON_API __declspec(dllimport)
#endif
#else
#define BCON_API
#endif
#endif
#define BCON_API __declspec(dllexport)
#ifdef __cplusplus
extern "C"{
#endif
    //outcode here
    BCON_API int N(int* target, int* result, int length, int thr_num);
    BCON_API int K(int* target, int* period, int* kstate, int* distance, int* counts, int length, int size, int thr_num = 1);
    BCON_API int R(int* target, int* result, int length, int size,int thr_num);
    BCON_API int X(int* target, int* result, int length, int size,int thr_num);
#ifdef __cplusplus
};
#endif

char Qlist[] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
          1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
          1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
          2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
          1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
          2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
          2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
          3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
          1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
          2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
          2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
          3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
          2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
          3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
          3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
          4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8 };     // set of 8 bit

unsigned char Rlist[256] = {0,};

bool BIGSIZE = false;

int translate(int size, int n)
{
    return (n >> (size - 1))* ((1) - (1 << size)) + 2 * n;
};

int rtranslate(int size, int n)
{
    return ((n & 1) <<(size - 1)) + (n >> 1);
}

int par_num(int x)
{
    int n = 0;
    n += Qlist[x & 255];
    n += Qlist[(x >> 8) & 255];
    n += Qlist[(x >> 16) & 255];
    n += Qlist[(x >> 24) & 255];
    return n;
}

//! function which checks whether a given state is representative or not.
/*!function to check whether given state is smallest state
    based on translational symmetry or not.
    return is period(positive integer), or error(-1)*/
int check(int size, int x)
{
    int a;
    int i;
    for (i = 0; i < size; i++) {
        a = translate(size, x);
            if (a < x) {
                return -1;
            }
            else if (a == x) {
                break;
            }
        x = a;
    }
    return i + 1;
}





void period_thread(int* target, int* period, int order, int unit, int size){
  int ind;
  for (size_t i = 0; i < unit; i++) {
    ind=i*unit+order;
    period[ind] = check(size, target[ind]);
  }
}

void distance_thread(int* target, int* distance, int* kstate, int start, int stop, vector<int> queue, int p, int size){
  int ind;
  int sind = start;
  for (vector<int>::size_type i = 0; i < queue.size(); ++i) {
    ind = queue[i];
    int temp = target[ind];
    kstate[ind*(size+1)] = temp;
    for (size_t j = 0; j < p; j++) {
      kstate[ind*(size+1)+j+1] = temp;
      distance[sind] =temp;
      distance[sind+1] = j;
      sind+=2;
      temp = rtranslate(temp, size);
    }

  }
}

//! funtion with making state set with translational symmetry
BCON_API int K(int* target, int* period, int* kstate, int* distance, int* counts, int length, int size, int thr_num = 1){
  //period and kstate is initialized as -1.
  //period : same length with target
  //kstate : shape (length, size+1) matrix
  //distance : shape(length, 2) matrix (state, distance) map.
  //counts : number of states in momentum k sector, its length is same with size, initialized as 0.
  vector<thread> workers;
  int unit = length / thr_num;
  int remain = length % thr_num;
  //distribute work (period check)
  for (size_t i = 0; i < thr_num; i++) {
      workers.push_back(thread(period_thread, target, period, i, unit, size));
  }
  //and remain
  for (size_t i = 0; i < remain; i++) {
    int ind= thr_num*unit+i;
    period[ind] = check(size, target[ind]);
  }
  //gather thread
  for (size_t i = 0; i < thr_num; i++) {
      workers[i].join();
  }
  workers.clear();

  //sorting and counting states by period
  const int sysize = size;
  int pcounts[sysize] = {0,};
  vector<int> indices[sysize];
  for (size_t i = 0; i < length; i++) {
    if (period[i]>0){
      pcounts[period[i]-1]+=1;
      indices[period[i]-1].push_back(i);
      //kstate[chnum*(size+1)] = target[i];
      for (size_t k = 0; k < size; k++) {
        if ((period[i]*k)%size){
          counts[k]+=1;
        }
      }
    }
  }
  //distribute work by period.
  int chnum = 0;
  for (size_t p = 1; p < size+1; p++) {
    workers.push_back(thread(distance_thread,target, distance, kstate, chnum, chnum+pcounts[p-1], indices[p-1],p, size));
    chnum+= 2*p*pcounts[p-1];
  }
  //collect
  for (size_t i = 0; i < size; i++) {
    workers[i].join();
  }
  workers.clear();
  //return
  return length;
}


void Nthread(int* target, int* result, int min, int max) {
    for (size_t i = min; i < max; i++) {
        result[i] = par_num(target[i]);
    }
}

BCON_API int N(int* target, int* result, int length, int thr_num = 1) {
    vector<thread> workers;
    int unit = length / thr_num;
    int remain = length % thr_num;
    if (thr_num==1){ //single thread
      Nthread(target, result, 0,length);
    }
    else{
      //distribute work
      for (size_t i = 0; i < thr_num; i++) {
          workers.push_back(thread(Nthread, target, result, i * unit, (i + 1) * unit));
      }
      //and remain
      Nthread(target, result, thr_num * unit, thr_num * unit + remain);
      //gather thread
      for (size_t i = 0; i < thr_num; i++) {
          workers[i].join();
      }
    }
    workers.clear();
    return length;
}

void Xthread(int* target, int* result, int min, int max, int size) {
  int mask = (1<<size)-1;
    for (size_t i = min; i < max; i++) {
        result[i] = target[i]^mask;
    }
}

BCON_API int X(int* target, int* result, int length, int size, int thr_num = 1) {
  vector<thread> workers;
  int unit = length / thr_num;
  int remain = length % thr_num;
  if (thr_num==1){ //single thread
    Xthread(target, result, 0,length,size);
  }
  else{
    //distribute work
    for (size_t i = 0; i < thr_num; i++) {
        workers.push_back(thread(Xthread, target, result, i * unit, (i + 1) * unit, size));
    }
    //and remain
    Xthread(target, result, thr_num * unit, thr_num * unit + remain, size);
    //gather thread
    for (size_t i = 0; i < thr_num; i++) {
        workers[i].join();
    }
  }
  workers.clear();
  return length;
}

//! function which reverse the bit of state.
int reverse_init() {
    for (size_t s = 0; s < 256; s++) {
      for (size_t i = 0; i < 8; i++) {
        Rlist[s] += ((s >> i) & 1) << (7 - i);
      }
    }
    BIGSIZE = true;
}

//! function which reverse the bit of state.
int parity(int* target, int* result, int min, int max, int size) {
    int p = 0;
    for (size_t index = min; index < max; index++) {
      for (size_t i = 0; i < size; i++) {
          p += ((target[index] >> i) & 1) << (size - i);
      }
      result[index] = p;
    }
}

//! dynamic programming version of function `parity`.
void Rthread(int* target, int* result, int min, int max, int size) {
  int repeat = size/8;
  int remain = size%8;
  for (size_t i = min; i < max; i++) {
    int p = 0;         //bit reverse for remain;
    for (size_t j = 0; j < remain; j++) {
      p+= (target[i]>>(j)&1)<<(size-j);
    }
    int rest = (target[i]>>remain);  //use calculation of 8-bit sector.
    for (size_t j = 0; j < repeat; j++) {
      p+=Rlist[(rest>>(8*j))&255]<<(8*(repeat-j-1));
    }
    result[i] = p;
  }
}

BCON_API int R(int* target, int* result, int length, int size, int thr_num = 1) {
  vector<thread> workers;
  int unit = length / thr_num;
  int remain = length % thr_num;
  if (size>15){
    if (!BIGSIZE){ //initializing
      reverse_init();
    }

    if (thr_num==1){ //single thread
      Rthread(target, result, 0,length,size);
    }
    else{
      //distribute work
      for (size_t i = 0; i < thr_num; i++) {
          workers.push_back(thread(Rthread, target, result, i * unit, (i + 1) * unit, size));
      }
      //and remain
      Rthread(target, result, thr_num * unit, thr_num * unit + remain, size);
      //gather thread
      for (size_t i = 0; i < thr_num; i++) {
          workers[i].join();
      }
    }
    workers.clear();
  }
  else{
    if (thr_num==1){ //single thread
      parity(target, result, 0,length,size);
    }
    else{
      //distribute work
      for (size_t i = 0; i < thr_num; i++) {
          workers.push_back(thread(parity, target, result, i * unit, (i + 1) * unit, size));
      }
      //and remain
      parity(target, result, thr_num * unit, thr_num * unit + remain, size);
      //gather thread
      for (size_t i = 0; i < thr_num; i++) {
          workers[i].join();
      }
    }
    workers.clear();
  }
  return length;
}
