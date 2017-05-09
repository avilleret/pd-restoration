      # Makefile for mylib

      lib.name = restoration

      class.sources = declicker~.c

      declicker-adrinas~.class.sources = declicker-adrinas~.c adrinas.c
      #cflags = -fopenmp -O2 // TODO OpenMP optimization leads to "singular matrix" error
      #declicker-adrinas~.class.ldflags = -fopenmp -O2

      denoiser2~.class.sources =  nrepel.c estimate_noise_spectrum.c extra_functions.c spectral_processing.c denoise_gain.c
      denoiser2~.class.ldlibs += -lfftw3f
      denoiser2~.class.ldflags += -lfftw3f

      datafiles = declicker~-help.pd README.md LICENSE.txt

      include Makefile.pdlibbuilder
