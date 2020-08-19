#ifndef QDLDL_TYPES_H
# define QDLDL_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

#include <limits.h> //for the QDLDL_INT_TYPE_MAX

// QDLDL integer and float types

typedef long long    QDLDL_int;   /* for indices */
typedef double  QDLDL_float; /* for numerical values  */
typedef unsigned char   QDLDL_bool;  /* for boolean values  */

//Maximum value of the signed type QDLDL_int.
#define QDLDL_INT_MAX LLONG_MAX

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef QDLDL_TYPES_H */
