import pyqtgraph as pg
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyqtgraph.Qt import QtCore, QtGui

__all__ = ['DemoLUTWidget']


# Gradients is a global variable declared in
# pyqtgraph.graphicsItems.GradientEditorItem
#
Gradients['reversed_flame'] = {'mode': 'rgb',
                               'ticks': [(0.8,
                                          (7, 0, 220, 255)),
                                         (0.5,
                                          (236, 0, 134, 255)),
                                         (0.2,
                                          (246, 246, 0, 255)),
                                         (0.0,
                                          (255, 255, 255, 255)),
                                         (1.0,
                                          (0, 0, 0, 255))]}

Gradients['classic'] = {'mode': 'rgb',
                        'ticks': [(0.0,
                                   (255, 255, 255, 255)),
                                  (0.2,
                                   (65, 172, 160, 255)),
                                  (0.4,
                                   (65, 172, 160, 255)),
                                  (0.65,
                                   (0, 0, 127, 255)),
                                  (0.9,
                                   (255, 81, 0, 255)),
                                  (1.0,
                                   (255, 255, 0, 255))]}
Gradients['grey'] = {'mode': 'rgb',
                     'ticks': [(1.0,
                                (0, 0, 0, 255)),
                               (0.0,
                                (255, 255, 255, 255))]}

# delete some gradient options for the sake of reducing color options
#
del Gradients['bipolar']
del Gradients['flame']
del Gradients['spectrum']
del Gradients['greyclip']
del Gradients['cyclic']

#------------------------------------------------------------------------
#
# file: DemoLUTWidget
#
# this class holds the logic for the LUT widget contained in spectrogram
# preferences tab in DemoPreferencesWidget
#
class DemoLUTWidget(pg.GraphicsView):
    def __init__(self,
                 cfg_dict_a=None,
                 *args,
                 **kwargs):
        background = kwargs.get('background', 'default')
        # print "lut before pg.GraphicsView.__init__()"
        pg.GraphicsView.__init__(self,
                                 parent=None,
                                 useOpenGL=False,
                                 background=background)
        # print "lut after pg.GraphicsView.__init__()"
        self.item = DemoLUTItem(*args, **kwargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(QtGui.QSizePolicy.Preferred,
                           QtGui.QSizePolicy.Expanding)
        self.setMinimumWidth(95)
        self.fillHistogram(False)

        # set the initial levels of the plot
        # TODO: read from config file
        #
        self.config_dict = cfg_dict_a
        self.setLevels(int(self.config_dict['level_high']),
                       int(self.config_dict['level_low']))

    def sizeHint(self):
        return QtCore.QSize(115, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class DemoLUTItem(pg.HistogramLUTItem):
    def __init__(self):
        super(DemoLUTItem, self).__init__()
        self.image_list = []

        # load a default colormap
        #
        self.gradient.loadPreset('classic')

    def addImageItem(self, img):
        self.image_list.append(img)
        img.sigImageChanged.connect(self.imageChanged)

        # send function pointer, not the result
        #
        img.setLookupTable(self.getLookupTable)
        self.regionChanged()
        self.imageChanged(autoLevel=True)

    def gradientChanged(self):
        if self.image_list:
            if self.gradient.isLookupTrivial():
                for image in self.image_list:

                    # lambda x: x.astype(np.uint8))
                    #
                    image.setLookupTable(None)
            else:
                for image in self.image_list:

                    # send function pointer, not the result
                    #
                    image.setLookupTable(self.getLookupTable)
        self.lut = None
        self.sigLookupTableChanged.emit(self)
        # self.print_table()

    def print_table(self):
        ticks = []
        for t in self.gradient.ticks:
            c = t.color
            ticks.append((self.gradient.ticks[t],
                          (c.red(),
                           c.green(),
                           c.blue(),
                           c.alpha())))
        print (ticks)

    def regionChanging(self):
        if self.image_list:
            for image in self.image_list:
                image.setLevels(self.region.getRegion())
        self.sigLevelsChanged.emit(self)
        self.update()

    def imageChanged(self, autoLevel=False, autoRange=False):

        # this try block is here so that the demo will not show errors when the entire
        # plot window is at 0. This occurs often at the beginning of files
        #
        try:
            # update histogram, just using the 1st image as a reference
            #
            histogram = self.image_list[0].getHistogram()
            if histogram[0] is None:
                return
            self.plot.setData(*histogram)

            for image in self.image_list:
                image.setLevels(self.getLevels())
        except:
            pass
    def delete_image_array(self):
        self.image_array = []
