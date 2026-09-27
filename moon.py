#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp
import mrcal

# We're in a world moonaz-left-up frame
#
# The moon is above the horizon, at el. The moon frame is translated
# world, without rotation.
#
# We're at sunset: the sun is at z=0

Rmoon_orbit = 384.8e6
Rmoon       = 1.737e6

imagersize       = np.array((500,500))
fov_extra_factor = 2.



def sample_terminator(vsun,
                      pmoon,
                      Nsamples = 100):
    # maps to x,y,vsun frame
    R_to_aligned = mrcal.R_aligned_to_vector(vsun)

    th = np.linspace(0, 2.*np.pi, Nsamples)
    valigned = \
        nps.cat(np.cos(th),
                np.sin(th),
                np.zeros(th.shape),
                ).T
    return \
        pmoon + \
        Rmoon * \
        mrcal.rotate_point_R(R_to_aligned.T,
                             valigned)

def moon_outline(vmoon,
                 th_rad,
                 Nsamples = 100):

    # maps to x,y,vmoon frame
    R_to_aligned = mrcal.R_aligned_to_vector(vmoon)

    vedge = mrcal.rotate_point_r(R_to_aligned[0] * th_rad,
                                 vmoon)
    return \
        Rmoon * \
        mrcal.rotate_point_r( vmoon * nps.dummy(np.linspace(0, 2.*np.pi, Nsamples),
                                                -1),
                              vedge )


def fibonacci_sphere(n):
    """Nice-looking even-ish spaced points on the surface of a sphere

Reports n points that cover the whole sphere. This is a magical routine produced
by Claude, which it lifted from Stackoverflow:

  https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

I don't have anything to say about it that isn't on that page. It is magical.

    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    z = 1 - 2 * i/(n-1)

    # clip for roundoff
    r = np.sqrt(np.clip(1.0 - z**2, 0, 1))
    theta = golden_angle * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return nps.cat(x,y,z).T

def show(*,
         el_moon_deg,
         # 90 = half-moon illuminated
         az_sun_deg):

    el_moon = el_moon_deg * np.pi/180.


    # This should depend on the radius of the earth, since my coord system is
    # centered on the surface, NOT at the center. But since Rearth << Rmoon_orbit, I
    # ignore that here
    vmoon = \
        np.array(( np.cos(el_moon),
                   0,
                   np.sin(el_moon)))
    pmoon = Rmoon_orbit * vmoon

    vsun = np.array((np.cos(az_sun_deg * np.pi/180.),
                     np.sin(az_sun_deg * np.pi/180.),
                     0.))
    pterminator_mlu  = sample_terminator(vsun, pmoon)



    p_moon_outline = moon_outline(vmoon, Rmoon/Rmoon_orbit)


    vmoon_surface_samples = fibonacci_sphere(n = 1000)
    pmoon_surface_samples_mlu = vmoon_surface_samples * Rmoon + pmoon
    # facing the viewer and
    # illuminated
    mask_sample_visible = \
        (nps.inner(pmoon_surface_samples_mlu, vmoon_surface_samples) < 0) * \
        (nps.inner(vsun,                      vmoon_surface_samples) > 0)





    R_mlu_cam = np.array(((0,  np.sin(el_moon),  vmoon[0]),
                          (-1, 0,                vmoon[1]),
                          (0,  -np.cos(el_moon), vmoon[2])))


    Rt_mlu_cam = nps.glue(R_mlu_cam,
                          np.zeros((3,)),
                          axis = -2)
    Rt_cam_mlu = mrcal.invert_Rt(Rt_mlu_cam)

    fov = 2.*Rmoon/Rmoon_orbit * fov_extra_factor
    fxy = imagersize/fov

    model = mrcal.cameramodel(intrinsics = ('LENSMODEL_STEREOGRAPHIC',
                                            np.array((*fxy, *(imagersize-1)/2.))),
                              imagersize = imagersize)




    q_terminator = \
       mrcal.project(mrcal.transform_point_Rt(Rt_cam_mlu, pterminator_mlu),
                     *model.intrinsics())
    q_moon_outline = \
       mrcal.project(mrcal.transform_point_Rt(Rt_cam_mlu,p_moon_outline),
                     *model.intrinsics())
    q_sample_visible = \
        mrcal.project(mrcal.transform_point_Rt(Rt_cam_mlu, pmoon_surface_samples_mlu[mask_sample_visible]),
                      *model.intrinsics())

    plot_tuples = \
        (( q_terminator,
           dict(_with = 'lines')),
         ( q_sample_visible,
           dict(_with = 'points pt 7 ps 0.5')),
         ( q_moon_outline,
           dict(_with = 'lines')),
         )
    plot_kwargs = \
        dict( tuplesize = -2,
              unset  = ('xtics', 'ytics'),
              square = 'True',
              xrange = (0,imagersize[0]-1),
              yrange = (imagersize[1]-1, 0),
              title  = f'el_moon={el_moon_deg:.1f} az_sun={az_sun_deg:.1f}')

    return (*plot_tuples, plot_kwargs)


layout = (3,3)
N = layout[0]*layout[1]

el_moon_deg = 60.
az_sun_deg_all = np.linspace(0, 180, N)
plot_tuples_kwargs = \
    [ show(el_moon_deg = el_moon_deg,
           az_sun_deg  = az) \
      for az in az_sun_deg_all ]
gp.plot(*plot_tuples_kwargs,
        multiplot = f'title "Scanning sun azimuth at sunset at mid-moon-elevation" layout {layout[0]},{layout[1]}',
        hardcopy  = '/tmp/scan-sun-az.pdf',
        terminal  = 'pdfcairo noenhanced size 8in,8in')

az_sun_deg = 135.
el_moon_deg_all = np.linspace(0, 90, N)
plot_tuples_kwargs = \
    [ show(el_moon_deg = el,
           az_sun_deg  = az_sun_deg) \
      for el in el_moon_deg_all ]
gp.plot(*plot_tuples_kwargs,
        multiplot = f'title "Scanning moon elevation at sunset at 3/4 moon" layout {layout[0]},{layout[1]}',
        hardcopy  = '/tmp/scan-moon-el.pdf',
        terminal  = 'pdfcairo noenhanced size 8in,8in')

