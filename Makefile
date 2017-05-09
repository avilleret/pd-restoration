      # Makefile for mylib

      lib.name = restoration

      class.sources = declicker~.c

      declicker-adrinas~.class.sources = declicker-adrinas~.c adrinas.c
      #cflags = -fopenmp -O2 // TODO OpenMP optimization leads to "singular matrix" error
      #declicker-adrinas~.class.ldflags = -fopenmp -O2

      denoiser2~.class.sources =  nrepel.c estimate_noise_spectrum.c extra_functions.c spectral_processing.c denoise_gain.c
      define forLinux
          denoiser2~.class.ldlibs += -lfftw3f
      endef
      
      define forDarwin
          # WARNING : brew ships 64bit version of fftw so I build a FAT binary from source and link directly to it
          denoiser2~.class.ldlibs += /usr/local/lib/libfftw3f.a
      endef
      datafiles = declicker~-help.pd denoiser2~-help.pd README.md LICENSE.txt

      include Makefile.pdlibbuilder
