#include <stdio.h>
#include <stdlib.h>

int foo(int a, int b, int c) {
    if (a > b) {
        return a + c;
    } else {
        return b + c;
    }
    // return a + b + c;
}

int main(int argc, char *argv[]) {
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int c = foo(a, b, 7);
    printf("%d\n", c);
    return 0;
}