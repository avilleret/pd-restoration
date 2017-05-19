# restoration

Audio restoration library for pure-data.

- declicker~ is a port of Audacity declicker tool
- declicker-adrinas~ is an implementation of [Laurent Oudre's declicker](http://www.ipol.im/pub/art/2015/64/?utm_source=doi), based on his demonstration code
- denoiser2~ is a denoiser based on [noise-repellent](https://github.com/lucianodato/noise-repellent.git) LV2 plugin by Luciano Dato

It has been tested on Ubuntu, Debian and Mac OSX (both 32 and 64bit) with pd-extended 0.43 and pd vanille 0.47-1.

## building 

Build system is based on [pd-lib-builder](https://github.com/pure-data/pd-lib-builder). 

denoiser2~ relies on fftw3 library, you can install it on Debian based system with :

    sudo apt install fftw-dev

and on Mac OS with  :

    brew install fftw

At the time of writing this, brew's build of libfftw are 32 bit only. So if you want to use it with Pd 64bit, you have to rebuild it from source with appropriate flags. And to make deployment easier, I link statically to fftw, which produce bigger binary, but user don't have to install fftw.

