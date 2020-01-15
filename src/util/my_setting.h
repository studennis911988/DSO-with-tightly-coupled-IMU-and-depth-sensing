#ifndef MY_SETTING_H
#define MY_SETTING_H


#define USE_RGB    0    // use RGB or infra1 for image resourse
#define USE_INFR1  1

#define DEPTH_WEIGHT  1   // lamda factor in photometric error function

#define TRACE_ALL_ON_EPIPOLAR   0   // trace all immaturepoints on epipolar line including the points with depth matching

#define DEPTH_SCALE 0.001  // depth scale (1000 = 1m)

#define DEPTH_RANGE_MIN  0.11  // depth trust region range
#define DEPTH_RANGE_MAX  10.1

#define TRACE_CODE_MODE  0  // print

#endif // MY_SETTING_H
