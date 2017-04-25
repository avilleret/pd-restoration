      # Makefile for mylib

      lib.name = restoration

      class.sources = declicker~.c

      declicker-adrinas~.class.sources = declicker-adrinas~.c adrinas.c
      #cflags = -fopenmp -O2 // TODO OpenMP optimization leads to "singular matrix" error
      #declicker-adrinas~.class.ldflags = -fopenmp -O2

      datafiles = declicker~-help.pd README.md LICENSE.txt

      include Makefile.pdlibbuilder
