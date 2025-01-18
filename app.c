#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include <raylib.h>
#define RAYGUI_IMPLEMENTATION
#include "raygui-4.0/src/raygui.h"

// debug macros
#define str(x) #x
#define showint(x) printf(str(x)" = %d\n", x)
#define showuint64(x) printf(str(x)" = %llu\n", x)
#define showfloat(x) printf(str(x)" = %f\n", (double)x)
#define showhex(x) printf(str(x)" = 0x%x\n", x)
#define showptr(x) printf(str(x)" = %p\n", x)
#define showaddr(x) printf("&" str(x)" = %p\n", &x)

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define min3(a,b,c) min(a, min(b, c))
#define max3(a,b,c) max(a, max(b, c))
#define clamp(x,a,b) min(max(x, a), b);

#define allocate(n, type) ((type *) calloc(n, sizeof(type)))

#define for_i(a,b) for (int i = a; i < b; i += 1)
#define for_j(a,b) for (int j = a; j < b; j += 1)
#define for_k(a,b) for (int k = a; k < b; k += 1)

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef float f32;
typedef double f64;


typedef struct
{
   f32 elems[3];
} Vec3F32;


typedef struct
{
   f32 elems[3][3];
} Mat3x3F32;


void printVec3F32(Vec3F32 x)
{
   puts("3-Vector");
   for_i(0, 3)
   {
      printf("%f ", x.elems[i]);
   }
   printf("\n");
}

void printMat3x3F32(Mat3x3F32 A)
{
   puts("3x3 Matrix");
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         printf("%f ", A.elems[i][j]);
      }
      printf("\n");
   }
}

Vec3F32 mul_sv(f32 s, Vec3F32 x)
{
   Vec3F32 result = {
      s * x.elems[0],
      s * x.elems[1],
      s * x.elems[2],
   };
   return result;
}

Vec3F32 add_vv(Vec3F32 a, Vec3F32 b)
{
   Vec3F32 result = {
      a.elems[0] + b.elems[0],
      a.elems[1] + b.elems[1],
      a.elems[2] + b.elems[2],
   };
   return result;
}

Vec3F32 sub_vv(Vec3F32 a, Vec3F32 b)
{
   Vec3F32 result = {
      a.elems[0] - b.elems[0],
      a.elems[1] - b.elems[1],
      a.elems[2] - b.elems[2],
   };
   return result;
}

f32 dot(Vec3F32 a, Vec3F32 b)
{
   return
      a.elems[0] * b.elems[0] +
      a.elems[1] * b.elems[1] +
      a.elems[2] * b.elems[2];
}

f32 norm(Vec3F32 x)
{
   return sqrtf(dot(x, x));
}

Vec3F32 normalize(Vec3F32 x)
{
   f32 norm_x = norm(x);
   assert(norm_x > 0);
   return mul_sv(1.0 / norm_x, x);
}

Vec3F32 proj(Vec3F32 u, Vec3F32 a)
{
   return mul_sv(dot(u, a) / dot(u, u), u);
}

Vec3F32 getcol(Mat3x3F32 A, int j)
{
   assert(j >= 0);
   assert(j <= 2);
   Vec3F32 result = {
      A.elems[0][j],
      A.elems[1][j],
      A.elems[2][j],
   };
   return result;
}

Vec3F32 getrow(Mat3x3F32 A, int i)
{
   assert(i >= 0);
   assert(i <= 2);
   Vec3F32 result = {
      A.elems[i][0],
      A.elems[i][1],
      A.elems[i][2],
   };
   return result;
}

Vec3F32 lerp_vv(Vec3F32 a, Vec3F32 b, f32 t)
{
   assert(t >= 0);
   assert(t <= 1);
   return add_vv(mul_sv(1-t, a), mul_sv(t, b));
}

Mat3x3F32 fromrows(Vec3F32 r1, Vec3F32 r2, Vec3F32 r3)
{
   Mat3x3F32 result = {
      r1.elems[0], r1.elems[1], r1.elems[2],
      r2.elems[0], r2.elems[1], r2.elems[2],
      r3.elems[0], r3.elems[1], r3.elems[2],
   };
   return result;
}

Mat3x3F32 Identity3()
{
   Mat3x3F32 result = {
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
   };
   return result;
}

Mat3x3F32 Zeros3()
{
   Mat3x3F32 result = {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
   };
   return result;
}

Vec3F32 mul_mv(Mat3x3F32 A, Vec3F32 x)
{
   Vec3F32 y = {0};
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         y.elems[i] += A.elems[i][j] * x.elems[j];
      }
   }
   return y;
}

Mat3x3F32 mul_mm(Mat3x3F32 A, Mat3x3F32 B)
{
   Mat3x3F32 C = {0};
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         for_k(0, 3)
         {
            C.elems[i][j] += A.elems[i][k] * B.elems[k][j];
         }
      }
   }
   return C;
}

Mat3x3F32 mul_mm3(Mat3x3F32 A, Mat3x3F32 B, Mat3x3F32 C)
{
   return mul_mm(mul_mm(A, B), C);
}

Mat3x3F32 add_mm(Mat3x3F32 A, Mat3x3F32 B)
{
   Mat3x3F32 C = {0};
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         C.elems[i][j] = A.elems[i][j] + B.elems[i][j];
      }
   }
   return C;
}

Mat3x3F32 add_mm3(Mat3x3F32 A, Mat3x3F32 B, Mat3x3F32 C)
{
   return add_mm(add_mm(A, B), C);
}

Mat3x3F32 sub_mm(Mat3x3F32 A, Mat3x3F32 B)
{
   Mat3x3F32 C = {0};
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         C.elems[i][j] = A.elems[i][j] - B.elems[i][j];
      }
   }
   return C;
}

Mat3x3F32 mul_sm(f32 s, Mat3x3F32 A)
{
   Mat3x3F32 C = {0};
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         C.elems[i][j] = s * A.elems[i][j];
      }
   }
   return C;
}

Mat3x3F32 transpose(Mat3x3F32 A)
{
   Mat3x3F32 C = {0};
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         C.elems[i][j] = A.elems[j][i];
      }
   }
   return C;
}

f32 p_norm(Mat3x3F32 A, f32 p)
{
   assert(p >= 1);
   f32 result = 0;
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         result += powf(fabsf(A.elems[i][j]), p);
      }
   }
   result = powf(result, 1.0f / p);
   return result;
}

f32 maxnorm(Mat3x3F32 A)
{
   f32 maxabselem = A.elems[0][0];
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         maxabselem = max(maxabselem, fabsf(A.elems[i][j]));
      }
   }
   return maxabselem;
}

f32 det(Mat3x3F32 A)
{
   f32 a = A.elems[0][0];
   f32 b = A.elems[0][1];
   f32 c = A.elems[0][2];
   f32 d = A.elems[1][0];
   f32 e = A.elems[1][1];
   f32 f = A.elems[1][2];
   f32 g = A.elems[2][0];
   f32 h = A.elems[2][1];
   f32 i = A.elems[2][2];
   return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;
}

f32 trace(Mat3x3F32 A)
{
   f32 result = 0;
   for_i(0, 3)
   {
      result += A.elems[i][i];
   }
   return result;
}

Vec3F32 cross(Vec3F32 a, Vec3F32 b)
{
   f32 a1 = a.elems[0];
   f32 a2 = a.elems[1];
   f32 a3 = a.elems[2];
   f32 b1 = b.elems[0];
   f32 b2 = b.elems[1];
   f32 b3 = b.elems[2];
   Vec3F32 result = {
      a2 * b3 - a3 * b2,
      a3 * b1 - a1 * b3,
      a1 * b2 - a2 * b1,
   };
   return result;
}

// https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
Mat3x3F32 inv(Mat3x3F32 A)
{
   f32 detA = det(A);
   assert(detA != 0);
   Vec3F32 c1 = getcol(A, 0);
   Vec3F32 c2 = getcol(A, 1);
   Vec3F32 c3 = getcol(A, 2);
   Vec3F32 r1 = cross(c2, c3);
   Vec3F32 r2 = cross(c3, c1);
   Vec3F32 r3 = cross(c1, c2);
   Mat3x3F32 result = mul_sm(1.0f/detA, fromrows(r1, r2, r3));
   return result;
}

bool eq_Vec3F32(Vec3F32 a, Vec3F32 b)
{
   return a.elems[0] == b.elems[0] && a.elems[1] == b.elems[1] && a.elems[2] == b.elems[2];
}

bool iszero_Vec3F32(Vec3F32 x)
{
   return eq_Vec3F32(x, (Vec3F32){ 0, 0, 0 });
}

bool eq_Mat3x3F32(Mat3x3F32 a, Mat3x3F32 b)
{
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         if (a.elems[i][j] != b.elems[i][j])
         {
            return false;
         }
      }
   }
   return true;
}

bool approx(f32 a, f32 b, f32 tol)
{
   return fabsf(a - b) <= tol;
}

bool approx_Vec3F32(Vec3F32 a, Vec3F32 b, f32 tol)
{
   for_i(0, 3)
   {
      if (! approx(a.elems[i], b.elems[i], tol))
      {
         return false;
      }
   }
   return true;
}

bool approx_Mat3x3F32(Mat3x3F32 a, Mat3x3F32 b, f32 tol)
{
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         if (! approx(a.elems[i][j], b.elems[i][j], tol))
         {
            return false;
         }
      }
   }
   return true;
}

bool isorthogonal(Mat3x3F32 A, f32 tol)
{
   Vec3F32 c1 = getcol(A, 0);
   Vec3F32 c2 = getcol(A, 1);
   Vec3F32 c3 = getcol(A, 2);
   bool result = true;
   result &= approx(dot(c1, c2), 0, tol);
   result &= approx(dot(c1, c3), 0, tol);
   result &= approx(dot(c2, c3), 0, tol);
   result &= approx(norm(c1), 1.0, tol);
   result &= approx(norm(c2), 1.0, tol);
   result &= approx(norm(c3), 1.0, tol);
   return result;
}

bool isrotation_tol(Mat3x3F32 A, f32 tol)
{
   bool result = true;
   result &= isorthogonal(A, tol);
   result &= approx(det(A), 1.0, tol);
   return result;
}

bool isrotation(Mat3x3F32 A)
{
   f32 tol = 1e-5;
   return isrotation_tol(A, tol);
}

bool warn_ifnotrotation(Mat3x3F32 R)
{
   if (! isrotation(R))
   {
      puts("error: matrix is not a rotation matrix");
      printMat3x3F32(R);

      for (f32 tol = 1e-5; tol < 1e-2; tol *= 1e1)
      {
         if (isrotation_tol(R, tol))
         {
            printf("passes with a tolerance of %f", tol);
            break;
         }
      }
      return true;
   }
   else
   {
      return false;
   }
}

bool isskewsymmetric(Mat3x3F32 A)
{
   f32 tol = 1e-5;
   for_i(0, 3)
   {
      for_j(0, 3)
      {
         if (! approx(A.elems[i][j], -A.elems[j][i], tol))
         {
            return false;
         }
      }
   }
   return true;
}

Mat3x3F32 expm_powerseries(Mat3x3F32 A)
{
   Mat3x3F32 result = Identity3();
   Mat3x3F32 An = Identity3();
   f32 factorial = 1;
   for_i(1, 13)
   {
      An = mul_mm(An, A);
      factorial *= i;
      result = add_mm(result, mul_sm(1.0f / factorial, An));
   }
   return result;
}

Mat3x3F32 expm_scalingandsquaring(Mat3x3F32 A)
{
   f32 maxabselem = maxnorm(A);
   f32 n = ceilf(log2f(maxabselem));
   f32 m = exp2(n);
   // showfloat(n);
   // showfloat(m);

   Mat3x3F32 B = mul_sm(1/m, A);
   Mat3x3F32 result = expm_powerseries(B);
   for_i(0, n)
   {
      result = mul_mm(result, result);
   }
   return result;
}

Mat3x3F32 expm(Mat3x3F32 A)
{
   return expm_scalingandsquaring(A);
}

typedef struct
{
   Mat3x3F32 Q;
   Mat3x3F32 R;
} QR_Factorization;

// https://en.wikipedia.org/wiki/QR_decomposition#Using_the_Gram%E2%80%93Schmidt_process
QR_Factorization qr(Mat3x3F32 A)
{
   Vec3F32 a1 = getcol(A, 0);
   Vec3F32 a2 = getcol(A, 1);
   Vec3F32 a3 = getcol(A, 2);

   Vec3F32 u1 = a1;
   Vec3F32 e1 = normalize(u1);
   Vec3F32 u2 = sub_vv(a2, proj(u1, a2));
   Vec3F32 e2 = normalize(u2);
   Vec3F32 u3 = sub_vv(sub_vv(a3, proj(u1, a3)), proj(u2, a3));
   Vec3F32 e3 = normalize(u3);

   Mat3x3F32 Q = {
      e1.elems[0], e2.elems[0], e3.elems[0],
      e1.elems[1], e2.elems[1], e3.elems[1],
      e1.elems[2], e2.elems[2], e3.elems[2],
   };
   Mat3x3F32 R = {
      dot(e1, a1), dot(e1, a2), dot(e1, a3),
                0, dot(e2, a2), dot(e2, a3),
                0,           0, dot(e3, a3),
   };
   QR_Factorization result = { Q, R };
   return result;
}

f32 sgn(f32 x)
{
   if (x >= 0)
   {
      return 1;
   }
   else
   {
      return -1;
   }
}

typedef struct
{
   Mat3x3F32 T; // upper Hessenberg
   Mat3x3F32 Z; // orthogonal
} Schur_Factorization; // A = Z * T * Z'

Schur_Factorization schur(Mat3x3F32 A)
{
   // https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
   int max_iters = 100;
   f32 tol = 1e-7;

   Mat3x3F32 T = A;
   Mat3x3F32 Z = Identity3();
   for_i(0, max_iters)
   {
      if (fabsf(T.elems[2][0]) <= tol)
      {
         break;
      }

      // https://math.stackexchange.com/a/1262366
      f32 A21_squared = A.elems[2][1] * A.elems[2][1];
      f32 sigma = 0.5 * (A.elems[1][1] - A.elems[2][2]);
      f32 mu = A.elems[2][2] - ((sgn(sigma) * A21_squared) / (fabsf(sigma) + sqrtf(sigma * sigma + A21_squared)));
      Mat3x3F32 shift = mul_sm(mu, Identity3());
      QR_Factorization F = qr(sub_mm(T, shift));
      T = add_mm(mul_mm(F.R, F.Q), shift);
      Z = mul_mm(Z, F.Q);

      if (i == max_iters - 1)
      {
         puts("warning: schur QR iterations did not converge");
      }
   }

   Schur_Factorization result = { T, Z };
   return result;
}

Mat3x3F32 logm_powerseries(Mat3x3F32 A)
{
   int max_iters = 10;
   Mat3x3F32 AminusI = sub_mm(A, Identity3());
   Mat3x3F32 AminusI_n = sub_mm(A, Identity3());
   Mat3x3F32 result = {0};
   f32 sign = 1;
   for_i(1, max_iters+1)
   {
      Mat3x3F32 curTerm = mul_sm(1.0f / (sign * i), AminusI_n);
      result = add_mm(result, curTerm);
      // printMat3x3F32(curTerm);
      // printMat3x3F32(result);

      AminusI_n = mul_mm(AminusI_n, AminusI);
      sign *= -1;
   }
   return result;
};

typedef struct
{
   Vec3F32 w;
} SkewsymMat3x3F32;

Mat3x3F32 fromSkew(SkewsymMat3x3F32 A)
{
   f32 x = A.w.elems[0];
   f32 y = A.w.elems[1];
   f32 z = A.w.elems[2];
   Mat3x3F32 result = {
      0, -z,  y,
      z,  0, -x,
     -y,  x,  0,
   };
   return result;
}

SkewsymMat3x3F32 skewFromMat(Mat3x3F32 A)
{
   // assert(isskewsymmetric(A));
   if (! isskewsymmetric(A))
   {
      puts("warning: matrix is not skew symmetric");
   }
   f32 x = A.elems[2][1];
   f32 y = A.elems[0][2];
   f32 z = A.elems[1][0];
   SkewsymMat3x3F32 result;
   result.w.elems[0] = x;
   result.w.elems[1] = y;
   result.w.elems[2] = z;
   return result;
}

void printSkewMatrix(SkewsymMat3x3F32 S)
{
   printMat3x3F32(fromSkew(S));
}

Mat3x3F32 expm_Rodrigues(SkewsymMat3x3F32 S)
{
   f32 theta = norm(S.w);
   if (theta == 0)
   {
      return Identity3();
   }
   SkewsymMat3x3F32 unitS = { normalize(S.w) };
   Mat3x3F32 A = fromSkew(unitS);
   // R = I + sin(theta) * A + (1-cos(theta)) * A^2;
   Mat3x3F32 A2 = mul_mm(A, A);
   Mat3x3F32 term1 = Identity3();
   Mat3x3F32 term2 = mul_sm(sinf(theta), A);
   Mat3x3F32 term3 = mul_sm(1 - cosf(theta), A2);
   Mat3x3F32 result = add_mm3(term1, term2, term3);
   return result;
}

// principal logarithm
SkewsymMat3x3F32 logm_Rodrigues(Mat3x3F32 R)
{
   assert(isrotation(R));
   f32 trR = trace(R);
   f32 costheta = (trR - 1) / 2;
   costheta = clamp(costheta, -1, 1);
   f32 theta = acosf(costheta);

   if (theta == 0)
   {
      Mat3x3F32 result = Zeros3();
      return skewFromMat(result);
   }
   else if (approx(theta, M_PI, 0.01))
   {
      Mat3x3F32 A = sub_mm(R, Identity3());
      Vec3F32 nonzero_cols[3];
      int num_nonzero = 0;
      for_i(0, 3)
      {
         Vec3F32 c = getcol(A, i);
         if (! iszero_Vec3F32(c))
         {
            nonzero_cols[num_nonzero] = c;
            num_nonzero += 1;
         }
      }
      assert(num_nonzero >= 2);
      Vec3F32 c1 = nonzero_cols[0];
      Vec3F32 c2 = nonzero_cols[1];
      Vec3F32 rotation_axis = cross(c1, c2);
      rotation_axis = normalize(rotation_axis);
      SkewsymMat3x3F32 result;
      result.w = mul_sv(theta, rotation_axis);
      return result;
   }
   else
   {
      // https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
      // doesn't work when R is symmetric, which is also when theta = pi
      f32 scalefactor = theta / (2 * sinf(theta));
      Mat3x3F32 S = sub_mm(R, transpose(R));
      Mat3x3F32 result = mul_sm(scalefactor, S);
      return skewFromMat(result);
   }
}

// reference paper: A micro Lie theory for state estimation in robotics

SkewsymMat3x3F32 hat(Vec3F32 tau)
{
   SkewsymMat3x3F32 result = { tau };
   return result;
}

Vec3F32 vee(SkewsymMat3x3F32 S)
{
   return S.w;
}

Mat3x3F32 Exp(Vec3F32 tau)
{
   return expm_Rodrigues(hat(tau));
}

Vec3F32 Log(Mat3x3F32 R)
{
   return vee(logm_Rodrigues(R));
}

Vec3F32 closest_tau(Vec3F32 previoustau, Vec3F32 tau)
{
   Vec3F32 a = previoustau;
   Vec3F32 b = tau;

   f32 tol = 1e-3;
   f32 dist = norm(sub_vv(a, b));
   f32 err = remainder(dist, 2 * M_PI);
   // assert(approx(err, 0, tol));
   // if (! approx(err, 0, tol))
   // {
      // printVec3F32(a);
      // printVec3F32(b);
      // showfloat(dist);
      // showfloat(dist / 2 * M_PI);
      // showfloat(err);
   // }

   if (approx_Vec3F32(b, (Vec3F32){0,0,0}, tol))
   {
      return a;
   }
   else
   {
      int n = sgn(dot(a, b)) * roundf(dist / (2 * M_PI));
      Vec3F32 bn = normalize(b);
      Vec3F32 result = add_vv(b, mul_sv(2 * M_PI * n, bn));
      return result;
   }
}

Mat3x3F32 right_plus(Mat3x3F32 X, Vec3F32 Xlocal_tau)
{
   assert(isrotation(X));
   // warn_ifnotrotation(X);
   return mul_mm(X, Exp(Xlocal_tau));
}

Mat3x3F32 left_plus(Vec3F32 Xglobal_tau, Mat3x3F32 X)
{
   assert(isrotation(X));
   // warn_ifnotrotation(X);
   return mul_mm(Exp(Xglobal_tau), X);
}

Vec3F32 right_minus(Mat3x3F32 Y, Mat3x3F32 X)
{
   assert(isrotation(Y));
   assert(isrotation(X));
   return Log(mul_mm(inv(X), Y));
}

void tests()
{
   {
      Mat3x3F32 A = {
         1, 0, 0,
         0, 2, 0,
         0, 0, 3,
      };
      Vec3F32 x = { 1, 2, 3 };
      Vec3F32 y = mul_mv(A, x);
      // printVec3F32(y);
      assert(eq_Vec3F32(y, (Vec3F32){1, 4, 9}));
   }

   {
      Mat3x3F32 A = {
         -0.60330206, 0.37892473, 0.28918108,
         -1.3948197, 0.004168335, 0.83900154,
         0.052024767, -0.9670384, -0.9599657,
      };
      Mat3x3F32 B = {
         0.84627616, -0.115342505, 1.053899,
         0.28919032, 2.244182, -0.43674648,
         -1.1647018, -2.2986407, -0.71756023,
      };
      Mat3x3F32 AB_ans = {
         -0.73778856, 0.25523907, -1.0088184,
         -2.156384, -1.7583266, -2.073854,
         0.882443, 0.030405045, 1.1660128,
      };
      Mat3x3F32 AB = mul_mm(A, B);
      // printMat3x3F32(AB);
      assert(eq_Mat3x3F32(AB, AB_ans));

      Mat3x3F32 AplusB_ans = {
         0.2429741, 0.26358223, 1.3430802,
         -1.1056294, 2.2483504, 0.40225506,
         -1.1126771, -3.2656791, -1.677526,
      };
      Mat3x3F32 AplusB = add_mm(A, B);
      // printMat3x3F32(AplusB);
      assert(eq_Mat3x3F32(AplusB, AplusB_ans));

      f32 s = 3;
      Mat3x3F32 sA_ans = {
         -1.8099062, 1.1367742, 0.8675432,
         -4.184459, 0.012505004, 2.5170045,
         0.1560743, -2.9011152, -2.879897,
      };
      Mat3x3F32 sA = mul_sm(s, A);
      // printMat3x3F32(sA);
      assert(eq_Mat3x3F32(sA, sA_ans));

      Mat3x3F32 expA_ans = {
         0.4265973, 0.15702394, 0.18850186,
         -0.84604615, 0.57552314, 0.3249965,
         0.38513762, -0.4956509, 0.22382571,
      };

      Mat3x3F32 expA = expm_powerseries(A);
      // printMat3x3F32(expA);
      assert(approx_Mat3x3F32(expA, expA_ans, 1e-6));

      f32 p1 = 1.0;
      f32 p2 = 0.2;
      f32 p3 = 5.5;
      Mat3x3F32 S = {
          0,  p1,  p2,
        -p1,   0,  p3,
        -p2, -p3,   0,
      };
      f32 t = 600;
      Mat3x3F32 exptS_ans = {
        0.983928 ,  0.136021, 0.115417,
       -0.16995  ,  0.518034, 0.838337,
        0.0542231, -0.844505, 0.532838,
      };
      // Mat3x3F32 exptS_ps = expm_powerseries(mul_sm(t, S));
      // printMat3x3F32(exptS_ps);
      Mat3x3F32 exptS_ss = expm_scalingandsquaring(mul_sm(t, S));
      // printMat3x3F32(exptS_ss);
      assert(approx_Mat3x3F32(exptS_ss, exptS_ans, 1e-3));

      Mat3x3F32 S_transpose = transpose(S);
      Mat3x3F32 minus_S = mul_sm(-1, S);
      // printMat3x3F32(S_transpose);
      // printMat3x3F32(minus_S);
      assert(eq_Mat3x3F32(S_transpose, minus_S));
   }

   {
      Mat3x3F32 A = {
         -0.117684, 0.384348 ,  1.7328,
         -0.667926, 0.657683 , -0.0959416,
         -1.3195  , 0.0558083,  0.474272,
      };
      Mat3x3F32 Q_ans = {
         -0.0793235,  0.533319,  0.842187,
         -0.450208 ,  0.734612, -0.507601,
         -0.889394 , -0.419424,  0.181832,
      };
      Mat3x3F32 R_ans = {
         1.4836, -0.376217, -0.516073,
         0.0   ,  0.664715,  0.654735,
         0.0   ,  0.0     ,  1.59428,
      };

      QR_Factorization F = qr(A);
      // printMat3x3F32(A);
      // printMat3x3F32(F.Q);
      // printMat3x3F32(F.R);
      // printMat3x3F32(mul_mm(F.Q, F.R));
      assert(approx_Mat3x3F32(F.Q, Q_ans, 1e-6));
      assert(approx_Mat3x3F32(F.R, R_ans, 1e-5));
      assert(approx_Mat3x3F32(mul_mm(F.Q, F.R), A, 1e-6));
   }

   {
      Mat3x3F32 A = {
         -0.117684, 0.384348 ,  1.7328,
         -0.667926, 0.657683 , -0.0959416,
         -1.3195  , 0.0558083,  0.474272,
      };
      Schur_Factorization F = schur(A);
      // printMat3x3F32(F.T);
      // printMat3x3F32(F.Z);
      // printMat3x3F32(mul_mm3(F.Z, F.T, transpose(F.Z)));
      assert(approx_Mat3x3F32(mul_mm3(F.Z, F.T, transpose(F.Z)), A, 1e-5));
      assert(approx(F.T.elems[2][0], 0, 1e-7));
      assert(isorthogonal(F.Z, 1e-6));
   }

   {
      Mat3x3F32 A = {
         1, 2, 3,
         4, 5, 6,
         7, 8, 9,
      };

      f32 frobnorm = p_norm(A, 2);
      f32 frobnorm_ans = 16.881943016134134;
      assert(frobnorm == frobnorm_ans);
      f32 onenorm = p_norm(A, 1);
      f32 onenorm_ans = 45;
      assert(onenorm == onenorm_ans);
   }

   {
      Mat3x3F32 A = {
         -0.117684, 0.384348 ,  1.7328,
         -0.667926, 0.657683 , -0.0959416,
         -1.3195  , 0.0558083,  0.474272,
      };
      A = mul_sm(1e-1, A);
      Mat3x3F32 B = sub_mm(Identity3(), A);
      // printMat3x3F32(B);
      Mat3x3F32 logB = logm_powerseries(B);
      // printMat3x3F32(logB);
      Mat3x3F32 logB_ans = {
         0.0244934, -0.0384515  , -0.175382,
         0.0674225, -0.0654859  ,  0.0156237,
         0.133434 , -0.000941973, -0.0377556,
      };
      assert(approx_Mat3x3F32(logB, logB_ans, 1e-2));
      // printMat3x3F32(expm(logB));
   }

   {
      Mat3x3F32 A = {
         -0.117684, 0.384348 ,  1.7328,
         -0.667926, 0.657683 , -0.0959416,
         -1.3195  , 0.0558083,  0.474272,
      };
      f32 detA = det(A);
      assert(approx(detA, 1.572225893370977, 1e-6));
      f32 trA = trace(A);
      assert(approx(trA, 1.014271, 1e-6));
   }

   {
      Mat3x3F32 A = {
        1.34126 ,  0.969657, 0.365342,
       -0.536726,  0.256113, 0.186268,
       -0.431794, -0.149825, 0.251422,
      };
      Mat3x3F32 invA = inv(A);
      Mat3x3F32 ans = {
       0.374532, -1.21137 ,  0.353216,
       0.221211,  2.00849 , -1.80945,
       0.775047, -0.883531,  3.50573,
      };
      // printMat3x3F32(invA);
      assert(approx_Mat3x3F32(invA, ans, 1e-5));
   }

   {
      f32 t = 0.5;
      Mat3x3F32 R = {
         cosf(t), -sinf(t), 0,
         sinf(t),  cosf(t), 0,
         0,        0,       1,
      };
      assert(isrotation(R));
   }

   {
      Mat3x3F32 A = {
         0, -1,  0,
         1,  0,  0,
         0,  0,  0,
      };
      SkewsymMat3x3F32 S = skewFromMat(A);
      Mat3x3F32 R = expm_Rodrigues(S);
      Mat3x3F32 expA = {
          0.540302, -0.841471, 0.0,
          0.841471,  0.540302, 0.0,
          0.0     ,  0.0     , 1.0,
      };
      // printMat3x3F32(R);
      assert(approx_Mat3x3F32(R, expA, 1e-6));

      SkewsymMat3x3F32 S2 = logm_Rodrigues(R);
      Mat3x3F32 logR = fromSkew(S2);
      // printMat3x3F32(logR);
      assert(approx_Mat3x3F32(logR, A, 1e-6));
   }

   {
      SkewsymMat3x3F32 B;
      B.w.elems[0] =  0.377293;
      B.w.elems[1] = -0.3101;
      B.w.elems[2] = -0.608149;

      Mat3x3F32 expB = expm_Rodrigues(B);
      Mat3x3F32 ans = {
        0.778572, 0.492743, -0.388626,
       -0.603929, 0.756625, -0.250576,
        0.170575, 0.429794,  0.886669,
      };
      // printMat3x3F32(expB);
      assert(approx_Mat3x3F32(expB, ans, 1e-6));
      SkewsymMat3x3F32 S = logm_Rodrigues(expB);
      assert(approx_Vec3F32(S.w, B.w, 1e-6));
   }

   {
      Mat3x3F32 R = {
        -1,  0,  0,
         0, -1,  0,
         0,  0,  1
      };
      SkewsymMat3x3F32 S = logm_Rodrigues(R);
      // printSkewMatrix(S);
      // printMat3x3F32(expm_Rodrigues(S));
      assert(approx_Mat3x3F32(expm_Rodrigues(S), R, 1e-6));
   }

   {
      f32 perturb = 1e-2;
      f32 t = M_PI - perturb;
      Mat3x3F32 R = {
         cosf(t), -sinf(t), 0,
         sinf(t),  cosf(t), 0,
         0,        0,       1,
      };
      SkewsymMat3x3F32 S = logm_Rodrigues(R);
      Mat3x3F32 expS = expm_Rodrigues(S);
      // showfloat(t);
      // printMat3x3F32(expS);
      // printMat3x3F32(R);
      // showfloat(maxnorm(sub_mm(expS, R)));
      assert(approx_Mat3x3F32(expm_Rodrigues(S), R, 1e-3));
   }

   {
      // Mat3x3F32 expB = expm_Rodrigues(B);
      Mat3x3F32 R = {
        0.778572, 0.492743, -0.388626,
       -0.603929, 0.756625, -0.250576,
        0.170575, 0.429794,  0.886669,
      };
      Schur_Factorization F = schur(R);
      // printMat3x3F32(F.T);
      // printMat3x3F32(F.Z);

      // --- schur(R) from julia ---
      // Schur{Float64, Matrix{Float64}, Vector{ComplexF64}}
      // T factor:
      // 3×3 Matrix{Float64}:
      //  0.710933  -0.70326   -1.58867e-8
      //  0.70326    0.710933  -2.63854e-7
      //  0.0        0.0        1.0
      // Z factor:
      // 3×3 Matrix{Float64}:
      //  -0.804931  -0.34365   -0.483726
      //  -0.551885   0.733045   0.397578
      //  -0.217965  -0.586984   0.779706
      // eigenvalues:
      // 3-element Vector{ComplexF64}:
      //  0.7109329716740831 + 0.7032600107507736im
      //  0.7109329716740831 - 0.7032600107507736im
      //  1.0000000566518343 + 0.0im

      Mat3x3F32 T_ans = {
        0.710933,  0.70326 , -1.58867e-8,
       -0.70326 ,  0.710933, -2.63854e-7,
        0.0     ,  0.0     ,  1.0,
      };
      Mat3x3F32 Z_ans = {
       -0.34365 , -0.804931, -0.483726,
        0.733045, -0.551885,  0.397578,
       -0.586984, -0.217965,  0.779706,
      };

      assert(approx_Mat3x3F32(mul_mm3(F.Z, F.T, transpose(F.Z)), R, 1e-6));
      assert(approx_Mat3x3F32(F.T, T_ans, 1e-6));
      assert(isorthogonal(F.Z, 1e-5));
   }

   {
      Vec3F32 a = { M_PI, 0, 0 };
      Vec3F32 b = { -M_PI, 0, 0 };
      Vec3F32 c = closest_tau(a, b);
      // printVec3F32(c);
      assert(approx_Vec3F32(c, a, 1e-6));
   }

   {
      Vec3F32 a = { 0, 0, 0 };
      Vec3F32 b = { 0, 0, 0 };
      Vec3F32 c = closest_tau(a, b);
      // printVec3F32(c);
      assert(approx_Vec3F32(c, a, 1e-6));
   }

   {
      Vec3F32 a = { 2 * M_PI, 0, 0 };
      Vec3F32 b = { 0, 0, 0 };
      Vec3F32 c = closest_tau(a, b);
      // printVec3F32(c);
      assert(approx_Vec3F32(c, a, 1e-6));
   }

   {
      Vec3F32 a = { 15 * M_PI, 0, 0 };
      Vec3F32 b = { -M_PI, 0, 0 };
      Vec3F32 c = closest_tau(a, b);
      // printVec3F32(c);
      assert(approx_Vec3F32(c, a, 1e-6));
   }
}

Vector3 fromVec3F32(Vec3F32 v)
{
   Vector3 result = {
      v.elems[0],
      v.elems[1],
      v.elems[2],
   };
   return result;
}

Vec3F32 fromVector3(Vector3 v)
{
   Vec3F32 result = {
      v.x,
      v.y,
      v.z,
   };
   return result;
}

void DrawMatrix(Mat3x3F32 A, int x, int y)
{
   for_i(0, 3)
   {
      DrawText(TextFormat("%f %f %f", A.elems[i][0], A.elems[i][1], A.elems[i][2]),
               x, y + 20 * i, 20, DARKGRAY);
   }
}

Color lerp_color(Color c1, Color c2, f32 t)
{
   assert(t >= 0);
   assert(t <= 1);
   Color result = {
      (1-t) * c1.r + t * c2.r,
      (1-t) * c1.g + t * c2.g,
      (1-t) * c1.b + t * c2.b,
      (1-t) * c1.a + t * c2.a,
   };
   return result;
}

void DrawCylinderEx_gradient(Vector3 startPos, Vector3 endPos, float startRadius, float endRadius, int sides, Color color)
{
   int num_segments = 20;
   int n = num_segments;

   f32 dx = (endPos.x - startPos.x) / n;
   f32 dy = (endPos.y - startPos.y) / n;
   f32 dz = (endPos.z - startPos.z) / n;

   for_i(0, n-1)
   {
      Vector3 x0 = {
         startPos.x + dx * i,
         startPos.y + dy * i,
         startPos.z + dz * i,
      };
      Vector3 x1 = {
         startPos.x + dx * (i+1),
         startPos.y + dy * (i+1),
         startPos.z + dz * (i+1),
      };

      u8 base = 150;
      f32 s = ((f32)(i+1)) / n;
      Color c = {
         // color.r,
         // color.g,
         // color.b,
         base + s * (color.r - base),
         base + s * (color.g - base),
         base + s * (color.b - base),

         // base + s * (color.a - base),
         255,
      };
      DrawCylinderEx(x0, x1, startRadius, endRadius, sides, c);
   }
}

void DrawReferenceFrame_colors(Mat3x3F32 R, Color a1, Color a2, Color a3)
{
   // assert(isrotation(R));
   if(warn_ifnotrotation(R))
   {
      return;
   };
   Vector3 c1 = fromVec3F32(getcol(R, 0));
   Vector3 c2 = fromVec3F32(getcol(R, 1));
   Vector3 c3 = fromVec3F32(getcol(R, 2));
   Vector3 origin = {0,0,0};
   f32 r = 0.01;
   int n = 3; // number of sides
   // DrawCylinderEx(origin, c1, r, r, n, a1);
   // DrawCylinderEx(origin, c2, r, r, n, a2);
   // DrawCylinderEx(origin, c3, r, r, n, a3);
   DrawCylinderEx_gradient(origin, c1, r, r, n, a1);
   DrawCylinderEx_gradient(origin, c2, r, r, n, a2);
   DrawCylinderEx_gradient(origin, c3, r, r, n, a3);
}

void DrawReferenceFrame(Mat3x3F32 R)
{
   DrawReferenceFrame_colors(R, RED, GREEN, BLUE);
}

Vec3F32 mouse2tangent(Vec3F32 pointonsphere, Camera3D camera, Vector2 mousedelta)
{
   Vec3F32 up = fromVector3(camera.up);
   Vec3F32 x = cross(up, pointonsphere);
   Vec3F32 y = cross(pointonsphere, x);
   f32 sensitivity = 0.002;
   Vec3F32 dx = mul_sv(sensitivity * mousedelta.x, x);
   Vec3F32 dy = mul_sv(sensitivity * mousedelta.y * -1, y);
   Vec3F32 tangent = add_vv(dx, dy);
   DrawText(TextFormat("tangent x = %f", tangent.elems[0]), 20, 240, 20, DARKGRAY);
   DrawText(TextFormat("tangent y = %f", tangent.elems[1]), 20, 260, 20, DARKGRAY);
   return tangent;
}

Vec3F32 update_tau_from_keyboard(Vec3F32 tau_, f32 dt)
{
   Vector3 tau = fromVec3F32(tau_);
   f32 speed = 1;
   if (IsKeyPressed(KEY_R))
   {
      tau = (Vector3){ 0, 0, 0 };
   }
   if (IsKeyDown(KEY_RIGHT))
   {
      f32 dx = speed * dt;
      tau.x += dx;
   }
   if (IsKeyDown(KEY_LEFT))
   {
      f32 dx = -speed * dt;
      tau.x += dx;
   }
   if (IsKeyDown(KEY_UP))
   {
      f32 dz = -speed * dt;
      tau.z += dz;
   }
   if (IsKeyDown(KEY_DOWN))
   {
      f32 dz = speed * dt;
      tau.z += dz;
   }
   if (IsKeyDown(KEY_SPACE))
   {
      f32 dy = speed * dt;
      tau.y += dy;
   }
   if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT))
   {
      f32 dy = -speed * dt;
      tau.y += dy;
   }
   return fromVector3(tau);
}

int screenwidth;
int screenheight;

int main()
{
   tests();
   // return 0;

   SetConfigFlags(FLAG_WINDOW_RESIZABLE);
   InitWindow(screenwidth, screenheight, "");
   SetTargetFPS(60);
   int w = GetMonitorWidth(GetCurrentMonitor());
   int h = GetMonitorHeight(GetCurrentMonitor());
   screenwidth = w * 3 / 4;
   screenheight = h * 7 / 8;
   SetWindowSize(screenwidth, screenheight);
   SetWindowPosition(w / 8, h / 16);

   Camera camera = { 0 };
   camera.position = (Vector3){ 2.0f, 3.0f, 4.0f };    // Camera position
   camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
   camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
   camera.fovy = 60.0f;                                // Camera field-of-view Y
   camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type

   f32 t_framestart = GetTime();
   f32 t_framestart_prev = t_framestart;
   f32 t = 0;

   bool button_was_clicked = false;

   bool playing_animation = false;
   f32 t_animate = 0;

   Vec3F32 tau = { 0, 0, 0 };
   Vec3F32 tau_prev_frame = tau;

   bool sphere_is_being_dragged = false;
   Vector2 mouse_dragstart;
   Vec3F32 oldtau;
   Mat3x3F32 oldR;

   while (!WindowShouldClose())
   {
      t_framestart_prev = t_framestart;
      t_framestart = GetTime();
      f32 dt = t_framestart - t_framestart_prev;
      t += dt;

      screenwidth = GetScreenWidth();
      screenheight = GetScreenHeight();

      tau = update_tau_from_keyboard(tau, dt);
      Mat3x3F32 R = Exp(tau);

      if (! sphere_is_being_dragged && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
      {
         sphere_is_being_dragged = true;
         mouse_dragstart = GetMousePosition();
         oldtau = tau;
         oldR = R;
      }

      if (sphere_is_being_dragged)
      {
         if (IsMouseButtonDown(MOUSE_LEFT_BUTTON))
         {
            Vector2 mouse_dragend = GetMousePosition();
            Vector2 mousedelta = {
               mouse_dragend.x - mouse_dragstart.x,
               mouse_dragend.y - mouse_dragstart.y,
            };
            DrawText(TextFormat("dx = %f", mousedelta.x), 20, 200, 20, DARKGRAY);
            DrawText(TextFormat("dy = %f", mousedelta.y), 20, 220, 20, DARKGRAY);
            Vec3F32 pointonsphere = normalize(fromVector3(camera.position));
            Vec3F32 tangent = mouse2tangent(pointonsphere, camera, mousedelta);
            Vec3F32 tau_increment = cross(pointonsphere, tangent);
            R = left_plus(tau_increment, oldR);
            tau = Log(R);

            f32 dist = norm(sub_vv(tau_prev_frame, tau));
            if (dist > M_PI)
            {
               tau = closest_tau(tau_prev_frame, tau);
            }
         }
         else
         {
            sphere_is_being_dragged = false;
         }
      }

      tau_prev_frame = tau;

      if (button_was_clicked)
      {
         playing_animation = true;
         t_animate = 0;
      }

      Vec3F32 targettau = { -1, 0, 1 };
      Mat3x3F32 targetR = Exp(targettau);

      if (playing_animation)
      {
         t_animate += dt;
         if (t_animate >= 1)
         {
            playing_animation = false;
         }
         t_animate = clamp(t_animate, 0, 1);
      }

      Vec3F32 diff = right_minus(targetR, R);
      Vec3F32 v_animate = lerp_vv((Vec3F32){0,0,0}, diff, t_animate);
      Mat3x3F32 R_animate = right_plus(R, v_animate);
      Vec3F32 tau_animate = Log(R_animate);

      BeginDrawing();
      ClearBackground(RAYWHITE);

      {
         BeginMode3D(camera);

         DrawGrid(20, 1);

         f32 radius = 0.05;

         DrawReferenceFrame(R);
         DrawSphere(fromVec3F32(tau), radius, MAROON);

         DrawReferenceFrame(targetR);
         DrawSphere(fromVec3F32(targettau), radius/2, MAGENTA);

         if (playing_animation)
         {
            Color c1 = lerp_color(RED, ORANGE, t_animate);
            Color c2 = lerp_color(GREEN, LIME, t_animate);
            Color c3 = lerp_color(BLUE, SKYBLUE, t_animate);
            DrawReferenceFrame_colors(R_animate, c1, c2, c3);
            DrawSphere(fromVec3F32(tau_animate), radius/2, MAGENTA);
         }

         EndMode3D();
      }

      DrawText(TextFormat("ball position = %.3f, %.3f, %.3f", tau.elems[0], tau.elems[1], tau.elems[2]), 20, 60, 20, DARKGRAY);
      DrawMatrix(R, 20, 130);

      int ui_y = 100;
      Rectangle bounds = {20, ui_y, 100, 20};
      button_was_clicked = GuiButton(bounds, "Button");

      f32 t_frameend = GetTime();
      f32 frametime = 1000 * (t_frameend - t_framestart);
      DrawText(TextFormat("Frame time: %.3f ms", frametime), 20, 20, 20, DARKGRAY);

      EndDrawing();
   }

   return 0;
}
