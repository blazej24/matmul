/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki
 * Further modifications by Krzysztof Rzadca 2018
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>

int main(int argc, char * argv[])
{
    std::cout <<"test";
        struct timespec spec;

   MPI_Init(&argc,&argv); /* intialize the library with parameters caught by the runtime */

    clock_gettime(CLOCK_REALTIME, &spec);
    srand(spec.tv_nsec); // use nsec to have a different value across different processes

    unsigned t = rand() % 5;
    sleep(t);
    printf("Hello world from %d/%d (slept %u s)!\n", 0, 1, t);

    MPI_Finalize(); /* mark that we've finished communicating */

    return 0;
}
