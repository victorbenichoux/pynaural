from matplotlib.pyplot import *
import numpy as np
from scipy import stats
#import spatializer
from scipy.interpolate import interp1d
print 'spat plotutils'

def make_colorbar(c, cmap = cm.jet, ax = None, yticks = None):
    if ax is None:
        ax = gca()
    Nc = float(len(c))
    twinax = twinx(ax)
    cnorm = (c - c.min())/(c.max()-c.min())    
    for kc in xrange(len(c)):
        twinax.axhspan(kc/Nc*(c.max()-c.min()) + c.min(),(kc+1)/Nc*(c.max()-c.min()) + c.min(),color = cmap(cnorm[kc]))
    twinax.set_xticks([])
    twinax.set_ylim(c.min(), c.max())
    if not yticks is None:
        twinax.set_yticks(yticks)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_colored_line(x, y, **kwdargs):
    '''
    plot_colored_line(x, y, c=c, ...)
    '''
    # get color
    cmap = kwdargs.pop('cmap', cm.jet)
    c = kwdargs.pop('c', np.linspace(0, 1, x.shape[0]))
    ax = kwdargs.pop('ax', gca())
    Ncolor = kwdargs.pop('Ncolor', 1000)

    
    # all other kwd args are passed to plot

    # prepare interpolation of color
    c = (c - c.min())/(c.max()-c.min())
    cdense = interp1d(c, c, kind = 'cubic')
    xdense = interp1d(c, x, kind = 'cubic')    
    ydense = interp1d(c, y, kind = 'cubic')            

    cgrid = np.linspace(0, 1, Ncolor)
    for kp in xrange(Ncolor-1):
        curcgrid = np.linspace(cgrid[kp], cgrid[kp+1], 10)

        ax.plot(xdense(curcgrid), ydense(curcgrid), '-', color = cmap(cgrid[kp]), **kwdargs)
        #    plot(xdense(cgrid), ydense(cgrid))
def make_colorbar_circle(c, ax = None, linewidth = 10., Ncolor = 10):
    if ax is None:
        ax = gca()
    crad = np.mod(c/180.*np.pi+np.pi/2 + np.pi, 2*np.pi) - np.pi
    xx = np.cos(crad)
    yy = np.sin(crad)
    #    linecol = (crad - crad.min())/(crad.max()-crad.min())
    plot_colored_line(xx, yy, c=c, linewidth = linewidth, ax = ax, Ncolor = Ncolor)
    annotatefactor = 1.5
    factor2 = 0.05
    for kaz in xrange(len(c)):
        ax.annotate('%s' % (int(c[kaz])), (np.cos(crad[kaz])*annotatefactor, np.sin(crad[kaz])*annotatefactor))

    ax.set_xticks([])
    ax.set_yticks([])    
    totlim = 2.
    ax.set_xlim(-totlim, totlim)
    ax.set_ylim(-totlim, totlim)


def plot_distributions(grid, indata, **kwdargs):
    if not isinstance(indata, list):
        alldata = [indata]
    else:
        alldata = indata

    haslabels = False
    if 'labels' in kwdargs:
        haslabels = True
        labels = kwdargs.pop('labels')

    ax = kwdargs.pop('ax', gca())
    infill = kwdargs.pop('infill', False)

    circ_dist = kwdargs.pop('circular_distribution', False)
        
    allcurcolors = []
    iscircular = kwdargs.pop('circ', False)

    plot_hist = kwdargs.pop('plot_hist', False)
    if plot_hist:
        plot_distributions_histograms(grid, indata, **kwdargs)
        return
    
    nandata = []
    for kdata, data in enumerate(alldata):
        nandata.append(np.isnan(data))
        
        nout = len(np.nonzero(nandata[kdata])[0])
        n = data.size
        print 'DEBUG: plotutils.plot_distributions: %d points ignored (%.1f %%) in KDE estimation' % (nout, float(nout)/n*100)

        if haslabels:
            kwdargs['label'] = labels[kdata]
        
        curdata = data[-nandata[kdata]]
        if circ_dist:
            N = len(curdata)
            newdata = np.zeros(N*3)
            newdata[:N] = curdata
            newdata[N:2*N] = curdata+1.
            newdata[2*N:] = curdata-1.
            curdata = newdata

        ker = stats.kde.gaussian_kde(curdata)
        x = ax.plot(grid, ker(grid), **kwdargs)
        curcolor = x[-1]._color
        
        if haslabels:
            kwdargs.pop('label')

        kwdargs['alpha'] = .5
        kwdargs['color'] = curcolor
            
        if infill is True or infill is kdata:
            ax.fill_between(grid, ker(grid), **kwdargs)

        kwdargs.pop('color')
        kwdargs.pop('alpha')
        allcurcolors.append(curcolor)
        
    #all is plotted, ylims shouldn't change, so we can add mean and std bars
    ylims = list(ax.get_ylim())
    for kdata, data in enumerate(alldata):
        ymeanstd = (0.9 + .1 * float(kdata+1)/len(alldata)) * ylims[1]
        plot_mean_std(data[-nandata[kdata]], ymeanstd, allcurcolors[kdata], circ = iscircular, ax = ax)


    ylims[1] = ymeanstd*1.1
    if haslabels:
        legend()

    ax.set_ylim(ylims)
        
    ax.set_yticks([])
    ax.set_ylabel('probability')

    ax.set_xlim((grid[0], grid[-1]))
    

def plot_distributions_histograms(grid, indata, **kwdargs):
    xspan = (grid.min(), grid.max())
    ax = kwdargs.pop('ax', gca())    
    iscircular = kwdargs.pop('iscircular', False)    

    if not isinstance(indata, list):
        alldata = [indata]
    else:
        alldata = indata

    haslabels = False
    if 'labels' in kwdargs:
        haslabels = True
        labels = kwdargs.pop('labels')
        
    nandata = []
    for kdata, data in enumerate(alldata):
        nandata.append(np.isnan(data))
        alldata[kdata] = data[-nandata[kdata]]
        
    ax.hist(alldata, range = xspan, normed = True)
    
    colors = ['b', 'g', 'r']
    
    ylims = list(ax.get_ylim())
    for kdata, data in enumerate(alldata):
        ymeanstd = (0.9 + .1 * float(kdata+1)/len(alldata)) * ylims[1]
        plot_mean_std(data, ymeanstd, colors[kdata], circ = iscircular, ax = ax)

    ylims[1] = ymeanstd*1.1
    if haslabels:
        legend()

    ax.set_ylim(ylims)
        
    ax.set_yticks([])
    ax.set_ylabel('probability')

    ax.set_xlim((grid[0], grid[-1]))
        
    

def plot_mean_std(data, y, color, circ = False, ax = None):
    if ax is None:
        ax = gca()

    if not circ:
        mu = np.mean(data.flatten())
        sigma = np.std(data.flatten())
    
    else:
        #mu = mycircmean(data.flatten())
        #sigma = mycircstd(data.flatten())
        mu = stats.morestats.circmean(data.flatten(), high = .5, low = -.5)
        sigma = stats.morestats.circstd(data.flatten(), high = .5, low = -.5)
       
    print 'INFO: plotutils.plot_mean_std: mean = %f, std = %f' % (mu, sigma)
    ax.plot(mu, y, 'o', markersize = 10., color = color)
    
    startx, endx = mu - sigma/2, mu + sigma/2
    
    ax.plot(np.linspace(startx, endx, 10), y*np.ones(10), '-', linewidth = 3, color = color)


def plot_phases(f, ph, markerthing, ax = None):
    if ax is None:
        rc('grid', color='#316931', linewidth=1, linestyle='-')
        rc('xtick', labelsize=15)
        rc('ytick', labelsize=15)
        ax = axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='#d5de9c')
        
    polar(f, ph, markerthing, axes=ax)
    grid(True)
    
def plot_2D_distributions(xgrid, ygrid, datax, datay, **kwdargs):
    '''
    '''
    log_color = kwdargs.pop('log_color', False)

    ax = kwdargs.pop('ax', gca())
    alldata = np.vstack((datax, datay))

    nandata = np.isnan(alldata[0,:]) + np.isnan(alldata[1,:])
    nout = len(np.nonzero(nandata)[0])
    n = datax.size

    print 'DEBUG: plotutils.plot_distributions: %d points ignored (%.1f %%) in KDE estimation' % (nout, float(nout)/n*100)

    ker = stats.kde.gaussian_kde(alldata[:,-nandata])

    res = np.zeros((len(xgrid), len(ygrid)))

    for kx in range(len(xgrid)):
        curvals = np.vstack((xgrid[kx]*np.ones(len(ygrid)), ygrid))
        res[kx, :] = ker(curvals)
    
    if not log_color:
        res /= res.max()
        ax.pcolor(xgrid, ygrid, res.T, **kwdargs)
    else:
        ax.pcolor(xgrid, ygrid, np.log10(1+res.T), **kwdargs)

    ax.set_xlim(xgrid.min(), xgrid.max())
    ax.set_ylim(ygrid.min(), ygrid.max())

def plot_cp_cd_compare(blueCPs, blueCDs, greenCPs, greenCDs):
    for i in range(len(greenCDs)):
        plot(greenCPs[i], greenCDs[i], '.g', markersize = 10.)
        plot(blueCPs[i], blueCDs[i], '.b', markersize = 10.)
        
        if not np.abs(blueCPs[i] - greenCPs[i]) > 0.5:
            plot([blueCPs[i], greenCPs[i]], [blueCDs[i], greenCDs[i]], 'k-')
        else:
            if blueCPs[i] > greenCPs[i]:
                highCP = blueCPs[i]
                lowCP = greenCPs[i]
                highCD = blueCDs[i]
                lowCD = greenCDs[i]
            else:
                highCP = greenCPs[i]
                lowCP = blueCPs[i]
                highCD = greenCDs[i]
                lowCD = blueCDs[i]
                
            slope = (lowCD - highCD)/(lowCP + 1 - highCP)
            deltax = 0.5 - highCD
            midCD = highCP + deltax * slope

#            plot([highCP, lowCP+1], [highCD, midCD], 'k-')
#            plot([highCP-1, lowCP], [midCD, lowCD], 'k-')

#    plot(CPs[badcells], CDs[badcells], 'or')
    xlabel('CP')
    ylabel('CD')
    xlim(-0.5, 0.5)

def plot_CD_vs_CP_histograms(CPs, CDs, 
                             axlim = [-.5, .5, -1., 1.],
                             inset = False):

    if axlim != None:
        cpmin = axlim[0]
        cpmax = axlim[1]
        cdmin = axlim[2]
        cdmax = axlim[3]
    
    nbins = int(np.sqrt(len(CPs.flatten())))
    
    cpbins = np.linspace(cpmin, cpmax, nbins)
    cdbins = np.linspace(cdmin, cdmax, nbins)

    nullfmt   = NullFormatter()         # no labels

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histCPs = [left, bottom_h, width, 0.2]
    rect_histCDs = [left_h, bottom, 0.2, height]
    rect_MSEs = [left_h, bottom_h, 0.2, 0.2]

    axScatter = axes(rect_scatter)
    axHistCPs = axes(rect_histCPs)
    axHistCDs = axes(rect_histCDs)

    axMSEs = axes(rect_MSEs)
    axMSEs.set_yticks([])
    axMSEs.set_xticks([])

    if inset:
        CPgrid = np.linspace(axlim[0], axlim[1], 40)
        CDgrid = np.linspace(axlim[2], axlim[3], 40)

        values = np.zeros((len(CPgrid), len(CDgrid)))
        data = np.vstack((CPs.flatten(), CDs.flatten()))
        notnan = -np.isnan(CPs + CDs).flatten()
        print notnan.shape
        print data.shape
        kernel = stats.kde.gaussian_kde(data[:,notnan])
        for k,cp in enumerate(CPgrid):
            curgrid = np.vstack((cp * np.ones(len(CDgrid)), CDgrid))            
            values[k,:] = kernel.evaluate(curgrid)

        axIN = axes([left_h-0.23, bottom_h-0.23, 0.2, 0.2])
        axIN.contourf(CPgrid, CDgrid, values.T, 30, cmap = matplotlib.cm.jet)
        axIN.set_yticks([])
        axIN.set_xticks([])
    
    axScatter.scatter(CPs, CDs, color = 'b')
    print CPs.shape
    axHistCPs.hist(CPs, color = 'b', bins = cpbins)
    axHistCDs.hist(CDs, color = 'b', bins = cdbins, orientation='horizontal')

    axHistCPs.xaxis.set_major_formatter(nullfmt)
    axHistCDs.yaxis.set_major_formatter(nullfmt)
    axHistCDs.set_xticks([])
    axHistCPs.set_yticks([])
    
    axHistCPs.annotate('CPs histogram', (10, 100), xycoords='axes points')
    axHistCDs.annotate('CDs histogram', (10, 360), xycoords='axes points')


    axScatter.set_xlabel('CPs (cycles)')
    axScatter.set_ylabel('CDs (ms)')

    dataslope, dataintercept = spatializer.dsp.fit.circular_linear_regression(CPs, CDs)[:2]

    sumupstring = 'Fit:\n(slope): %.3f (intercept): %.3f;' % (dataslope, dataintercept)
    sumupstring += '\nNcells: %d' % len(CPs)

    axScatter.annotate(sumupstring, (10,10), xycoords='axes points')

    # Plotting linear regressions
    mx = np.min(CPs)
    Mx = np.max(CPs)

    xs = np.linspace(mx, Mx, 100)

    axScatter.plot(xs, dataslope*(np.mod(xs + .5 , 1) - .5) + dataintercept, '--k', label = 'fit', linewidth = 3.)

    axScatter.set_xlim(cpmin, cpmax)
    axScatter.set_ylim(cdmin, cdmax)

    axHistCPs.set_xlim(cpmin, cpmax)
    axHistCDs.set_ylim(cdmin, cdmax)
    return None


def do_cpcd_axis():
    xlim(-0.5, 0.5)
    ylim(-1, 1)
    xticks([-0.5, 0, 0.5])
    xlabel('CP (cyc)')
    ylabel('CD (ms)')

    
def fold_positions(azs):
    res = np.zeros_like(azs)
    for kaz, az in enumerate(azs):
        if az > 90:
            res[kaz] = 180-az
        elif az < -90:
            res[kaz] = -180-az
        else:
            res[kaz] = az
    return res
            
            


if __name__ == '__main__':
    if False:
        data = np.random.randn(1000)
        grid = np.linspace(-10, 20, 100)
        plot_distributions(grid, [data, (data*10)+10])
        
    if False:
        xgrid = np.linspace(-1, 3, 100)
        ygrid = np.linspace(-2, 3, 100)
        
        xdata = np.hstack((np.zeros(100), np.linspace(0, 1, 100)))
        ydata = np.hstack((np.linspace(2, 0, 100), np.zeros(100)))

        plot_2D_distributions(xgrid, ygrid, xdata, ydata, cmap = matplotlib.cm.binary)

        show()

    if False:
        cps0 = [0.2, 0.2, -0.2, -0.2]
        cds0 = [0, 0.5, 0, 0.5]
        cps1 = [-0.4, 0.4, 0.4, -0.4]
        cds1 = [1, -1, 1, -1]
        plot_cp_cd_compare(cps0, cds0, cps1, cps1)
        
        show()

    if True:
        from project_utils import *;from datastructures import *
        celldb = AnalysisDataset.load(DROPBOX_FOLDER+'cell_data.pkl').subset(lambda is_reference_measure, stim_level_spl : is_reference_measure and stim_level_spl == 60.)
        CPs = celldb.get('CP')
        CDs = celldb.get('CD')
        
        plot_CD_vs_CP_histograms(CPs, CDs, inset = True)
        savefigure('cpcdplane_withhistograms.pdf')
        show()
