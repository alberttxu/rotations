# Rotations using 3 parameters

There are many applications that use quaternions to represent 3D orientations and rotations, most notably games.
Although other parameterizations exist and are used in fields such as robotics, I haven't seen many educational examples written in pure c code.
Other online blog posts such as [this one](http://marctenbosch.com/quaternions/) overcomplicate things in my opinion, and moreover revert back to an implementation related to quaternions.
In this example I show that it is possible to compute 3D orientations and rotations entirely devoid of any math related to quaternions using only basic linear algebra theory and properties of matrices.

In short, any rotation matrix R can be equal to exp(S), where S is a skew-symmetric matrix.
And there are multiple values of S (infinite, in fact) that satisfy R = exp(S).
Out of the 9 entries in S, only 3 are unique, and these 3 parameters can be stored as a vector.
It turns out that said vector is an axis-angle representation (it points in a direction perpendicular to the plane of rotation, and it's norm is the amount of rotation in radians).
