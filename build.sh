#!/bin/sh

set -e

includes="
-I/opt/local/include
"

libraries="
-L/opt/local/lib
-lraylib
"

gcc -g3 -o app $includes app.c $libraries
