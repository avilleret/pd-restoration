/*------------------------------------------------------------------------------

  ADRINAS: Automatic Detection and Removal of Impulsive Noise in Audio Signals

  This code is part of the following publication and was subject
  to peer review:

    "Automatic Detection and Removal of Impulsive Noise in Audio Signals"
    by Laurent Oudre, Image Processing On Line, 2015.
    DOI:10.5201/ipol.2015.XXXXX
    http://dx.doi.org/10.5201/ipol.2015.XXXXX

  Copyright (c) 2012-2015 Laurent Oudre <laurent.oudre@cmla.ens-cachan.fr>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

------------------------------------------------------------------------------*/
#ifndef ADRINAS_HEADER
#define ADRINAS_HEADER

/*----------------------------------------------------------------------------*/
/* ADRINAS: Automatic Detection and Removal of Impulsive Noise in Audio Signals

   signal : input audio signal, an array of doubles
   N      : number of samples in the input signal
   p      : order of the AR model to be used
   K      : detection threshold
   Nw     : window size

   return value : pointer to a int signal of size N, containing burst detections
 */
int * adrinas(double * signal, int N, int p, double K, int b, int Nw);

#endif /* !ADRINAS_HEADER */
/*----------------------------------------------------------------------------*/
