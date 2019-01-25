import pickle
import os
import json

import numpy as np

#from pseudo_voigt_2d import pvoigt2d, update_params, make_models, Peak 
#from peak_deconvolution import Peak

def deconvPeaks(argServer):
    """ Main function that is called by CCPN

        
        Uses os.system to call "test_deconv_ccpn.py"
    """
    peaks = argServer.getCurrentPeaks()
    spectrum = "path"#argServer.
    #print(getNoiseEstimate(spectrum))
    peakList = []
    for pk in peaks:
        print(pk.sortedMeasurements())
        peakDic = {}
        name = pk.annotation.split()[0]
        print(name)
        position = [j.value for j in pk.sortedPeakDims()]
        print(position)
        peakDic["name"] = name
        for num,i in enumerate(position):
            dim = "w%d"%(num+1)
            peakDic[dim]=i
        pk = Peak(center_x=position[0],center_y=position[1],amplitude=1e5,assignment=name)
        print(pk)
        peakList.append(pk)
    with open("temp.pkl","w") as f:
        pickle.dump(peakList,f)
        #peakList.append(peakDic)

    os.system("./test_deconv_ccpn.py")

c_ind = 0
colors = ["#66ff33","#FF0000","#0066ff","#ff33cc","#ff9900","#33ccff","#9900cc"]

def addPeaks(argServer):
    from ccpnmr.analysis.core.Util import getAnalysisPeakList
    from ccpnmr.analysis.core.PeakBasic import pickPeak
    project = argServer.getProject()
    nmrProject = project.getCurrentNmrProject()
    exp = nmrProject.findFirstExperiment()
    spec = exp.findFirstDataSource()
    if spec.findFirstPeakList(name="fitted"):
        #print(spec.findFirstPeakList(name="fitted"))
        peakList = spec.findFirstPeakList(name="fitted")
        print("Adding peaks to %s"%peakList.name)
        
    else:
        
        peakList = spec.newPeakList(name="fitted")
        analysisPeakList = getAnalysisPeakList(peakList)
        analysisPeakList.symbolStyle = '+'
        analysisPeakList.symbolColor = colors[c_ind]
        analysisPeakList.textColor = colors[c_ind]
        print("Making new peaklist %s"%peakList.name)
        
    c_ind += 1
    c_ind = c_ind%(len(colors)-1)
#    json_peaks = json.load(open("final_peaks.json","r"))
#    for i in json_peaks:
#        #for pk in peakList:
#        # checks if peak has already been added to the fitted peak list
#        if type(i) is list:
#            # run through peaks in cluster
#            for j in i:
#                if j.get("in_fitted"):
#                    pass
#                else:
#                    pickPeak(peakList,[j["w2"],j["w1"],1.0],unit="ppm")
#                    j["in_fitted"] = True
#
#        elif i.get("in_fitted"):
#            pass
#        else:
#            pickPeak(peakList,[i["w2"],i["w1"],1.0],unit="ppm")
#            i["in_fitted"] = True
#    # This is inefficient... writing whole file again every time.
#    # Would be better to just update
#    json.dump(json_peaks,open("final_peaks.json","w"),sort_keys=True,separators=[",\n",":"])
##    print(json_peaks)
#    #newPeaks = project.newPeakList()
#    #print(newPeaks)

class Peak():


    def __init__(self,center_x,center_y,amplitude,prefix="",sigma_x=1.0,sigma_y=1.0,assignment="None"):
        """ Peak class 
            
            Data structure for nmrpeak
        """
        self.center_x = center_x
        self.center_y = center_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.amplitude = amplitude
        self.prefix = prefix 
        self.assignment = assignment
                      
    def param_dict(self):
        """ Make dict of parameter names using prefix """
        str_form = lambda x: "%s%s"%(self.prefix,str(x))
        par_dict = {str_form("center_x"):self.center_x,
                    str_form("center_y"):self.center_y,
                    str_form("sigma_x"):self.sigma_x,
                    str_form("sigma_y"):self.sigma_y,
                    str_form("amplitude"):self.amplitude,
                    str_form("fraction"):0.5}
        return par_dict

    def mask(self,data,r_x,r_y):
        # data is spectrum containing peak
        a, b = self.center_y, self.center_x
        n_y, n_x = data.shape
        y,x = np.ogrid[-a:n_y-a, -b:n_x-b]
        # create circular mask
        mask = x**2./r_x**2. + y**2./r_y**2. <= 1.0
        return mask


    def __str__(self):
        return "Peak: x=%.1f, y=%.1f, amp=%.1f, fraction=%.1f, prefix=%s, assignment=%s"%\
                (self.center_x,self.center_y,self.amplitude,0.5,self.prefix,self.assignment)


#from string import Template
#
#from memops.gui.Entry import Entry
#from memops.gui.FileSelect import FileType
#from memops.gui.FileSelectPopup import FileSelectPopup
#from memops.gui.Label import Label
#from memops.gui.PulldownList import PulldownList
#from memops.gui.ButtonList import ButtonList, UtilityButtonList
#from memops.gui.Button import Button
#from memops.gui.LabelFrame import LabelFrame
#from memops.gui.ScrolledMatrix import ScrolledMatrix
#from memops.gui.ObjectsTable import ObjectsTable
#from memops.gui.Text import Text
#from memops.gui.ScrolledText import ScrolledText
#from memops.gui.MessageReporter import showWarning, showOkCancel, showYesNo, showInfo
#from ccpnmr.analysis.popups.BasePopup import BasePopup
#from ccpnmr.analysis.core.PeakBasic import pickPeak
#from ccpnmr.analysis.core.ExperimentBasic import getDataDimIsotopes, getOnebondDataDims
#from ccpnmr.analysis.core.ExperimentBasic import getThroughSpacePeakLists
#from ccpnmr.analysis.core.Util import getAnalysisPeakList
# 
#
#TEMP_LIST_NAME = '__RadarSuperimposeTemp'
#READABLE = os.R_OK
# 
#def fitParamsMacro(argServer):
#    popup = Macro(argServer.parent,argServer)
#    popup.open()
#
#class Macro(BasePopup):
#
#    def __init__(self, parent, argServer, *args, **kw):
#        self.spectrum = None
#        self.dir = '.'
#        self.argServer = argServer
#        self.peakList = []
#        self.proc_dir = os.path.abspath("./")
#        
#        BasePopup.__init__(self, parent, *args, **kw)
#
#    def body(self, guiFrame):
#  
#        frame = LabelFrame(guiFrame, text='Options', grid=(0,0))
#        defFrame = LabelFrame(frame, text='Default parameters', grid=(1,0))
#        inputFrame = LabelFrame(frame, text='Input files',grid=(2,0))
#        generalFrame = LabelFrame(frame, text='General parameters',grid=(3,0))
#        peakFrame = LabelFrame(frame, text='Selected peaks', grid=(4,0))
#
#        # F1/F2
#        label = Label(defFrame, text='DEF_RADIUS_F1(ppm):', grid=(0,0))
#        self.F1_radEntry = Entry(defFrame, text=0.2, width=24, grid=(0,1))
#        label = Label(defFrame, text='DEF_RADIUS_F2(ppm):', grid=(1,0))
#        self.F2_radEntry = Entry(defFrame, text=0.02, width=24, grid=(1,1))
#        label = Label(defFrame, text='DEF_LINEWIDTH_F1(ppm):', grid=(0,2))
#        self.F1_lwEntry = Entry(defFrame, text=0.2, width=24, grid=(0,3))
#        label = Label(defFrame, text='DEF_LINEWIDTH_F2(ppm):', grid=(1,2))
#        self.F2_lwEntry = Entry(defFrame, text=0.02, width=24, grid=(1,3))
#
#        # ZCOOR
#        label = Label(inputFrame, text='ZCOOR:', grid=(0,0))
#        self.zcoorEntry = Entry(inputFrame, text='time_relax', width=24, grid=(0,3))
#        self.zcoorFile = Entry(inputFrame, text='procpar', width=60, grid=(0,1))
#        self.zcoorButton = Button(inputFrame, command=self.importZcoorFile,
#                        grid=(0,2), text='Choose File')
#
#        # SPECTRUM
#        label = Label(inputFrame, text='Spectrum:', grid=(1,0))
#        self.specFile = Entry(inputFrame, text='Spectrum', width=60, grid=(1,1))
#        self.specButton = Button(inputFrame, command=self.importSpectrumFile,
#                        grid=(1,2), text='Choose File')
#        label = Label(inputFrame, text='Output directory:', grid=(2,0))
#        self.out_dir = Entry(inputFrame, text='./fuda_analysis', width=60, grid=(2,1))
#
#        # NOISE
#        label = Label(generalFrame, text='Noise:', grid=(0,0))
#        self.noiseEntry = Entry(generalFrame, text=20000, width=24, grid=(0,1))
#        # DELAFACTOR
#        label = Label(generalFrame, text='DELAFACTOR:', grid=(0,2))
#        self.delayEntry = Entry(generalFrame, text=0.005, width=24, grid=(0,3))
#        # SHAPE 
#        label = Label(generalFrame, text='SHAPE:', grid=(0,4))
#        self.shapeEntry = Entry(generalFrame, text='GLORE', width=24, grid=(0,5))
#        # FITEXP 
#        label = Label(generalFrame, text='FITEXP:', grid=(1,0))
#        self.fitexpEntry = Entry(generalFrame, text='Y', width=2, grid=(1,1))
#        # VERB 
#        label = Label(generalFrame, text='VERBOSELEVEL:', grid=(1,2))
#        self.verbEntry = Entry(generalFrame, text='5', width=2, grid=(1,3))
#        # PRINT 
#        label = Label(generalFrame, text='PRINTDATA:', grid=(1,4))
#        self.printEntry = Entry(generalFrame, text='Y', width=2, grid=(1,5))
#        # LM 
#        label = Label(generalFrame, text='MAXFEV:', grid=(2,0))
#        self.maxfevEntry = Entry(generalFrame, text=500, width=5, grid=(2,1))
#        label = Label(generalFrame, text='TOL:', grid=(2,2))
#        self.tolEntry = Entry(generalFrame, text=1e-4, width=5, grid=(2,3))
#        # Isotope and baseline
#        label = Label(generalFrame, text='Isotope shift:', grid=(3,0))
#        self.isotopeshiftEntry = Entry(generalFrame, text="N", width=24, grid=(3,1))
#        label = Label(generalFrame, text='BASELINE:', grid=(3,2))
#        self.baseEntry = Entry(generalFrame, text="N", width=2, grid=(3,3))
#        
#
#        # Peak table
#        self.objects = []
#        self.paramFuDAtext = ""
#	self.paramFuDA = ScrolledText(peakFrame, width=100, height=10,
#	             text=self.paramFuDAtext, xscroll=False,grid=(0,0))
#        #attributes = ["serial","name","w1","w2","w1_rad","w2_rad"]
#        #attributeHeadings = {"serial":"Serial","name":"Assignment",
#        #        "w1":"F1ppm","w2":"F2ppm","w1_rad":"F1 radius (ppm)",
#        #        "w2_rad":"F2 radius (ppm)"}
#        #widgetClasses = {"serial":Text,"name":Text,"w1":Text,"w2":Text,
#        #        "w1_rad":Entry,"w2_rad":Entry}
#	#self.table = ObjectsTable(peakFrame, objects=self.objects,
#        #        widgetClasses=widgetClasses,attributes=attributes,
#        #        attributeHeadings=attributeHeadings,grid=(0,0))
#
#        updatePeaks = Button(peakFrame,command=self.peakTableUpdate,
#                text='Update params',grid=(1,0))
#        runFudaButton = Button(peakFrame,command=self.runFuDA,
#                text='runFuDA',grid=(1,1))
#
#	#self.textMatrix = []
#	#self.objectList = []
#	#self.table.update(objectList=self.objectList, textMatrix=self.textMatrix)
#
#    def importZcoorFile(self):
#  
#        fileTypes = [  FileType('varian', ['procpar']), FileType('All', ['*'])]
#        fileSelectPopup = FileSelectPopup(self, file_types=fileTypes, directory=self.dir,
#                            title='Import zcoor file', dismiss_text='Cancel',
#                            selected_file_must_exist=True, multiSelect=False,)
#
#        file = fileSelectPopup.getFile()
#        
#        self.proc_dir = fileSelectPopup.getDirectory()
#        self.zcoorFile.set(file)
#        #print(file)
#        #self.zcoorButton.set(file)
#
#    def importSpectrumFile(self):
#  
#        fileTypes = [  FileType('pipe', ['*.ft*','*.dat','*.DAT']), FileType('All', ['*'])]
#        fileSelectPopup = FileSelectPopup(self, file_types=fileTypes, directory=self.dir,
#                            title='Import spectrum', dismiss_text='Cancel',
#                            selected_file_must_exist=True, multiSelect=False,)
#
#        file = fileSelectPopup.getFile()
#        
#        self.spec_dir = fileSelectPopup.getDirectory()
#        self.specFile.set(file)
#
#    def peakTableUpdate(self):
#        peaks = self.argServer.getCurrentPeaks()
#	spectrum = "path"#argServer.
#	#print(getNoiseEstimate(spectrum))
#        overlapPeaks = "OVERLAP_PEAKS=(%s)\n"
#        olp_string = ""
#        notdefPeaks = ""
#        notdefPeak = "NOT_DEF_PEAK=(NAME=%s;RADIUS_F1=%.3f;RADIUS_F2=%.3f)\n"
#	for pk in peaks:
#    #        print(pk.sortedMeasurements())
#	    peakDic = {}
#	    name = pk.annotation.split()[0]
#    #        print(name)
#	    position = [j.value for j in pk.sortedPeakDims()]
#    #        print(position)
#	    peakDic["name"] = name
#	    for num,i in enumerate(position):
#		dim = "w%d"%(num+1)
#		peakDic[dim]=i
#	    #self.peakList.append(peakDic) 
#            #self.textMatrix.append([pk.serial,name,peakDic['w1'],peakDic['w2']])
#            #self.objectList.append(pk.serial)
#            self.peakList.append(P(pk.serial,name,peakDic['w1'],peakDic['w2'],0.2,0.2))
#            olp_string += "%s;"%name
#
#            _notdefPeak = notdefPeak%(name,float(self.F1_radEntry.get()),float(self.F2_radEntry.get()))
#            notdefPeaks+=_notdefPeak
#        # remove last semi colon
#        if len(peaks)>0:
#            olp_string = olp_string[:-1]
#            overlapPeaks = overlapPeaks%olp_string
#            hashline="#---------------------------------------------------------------------#\n"
#            self.paramFuDAtext+=overlapPeaks+notdefPeaks+hashline
#            self.paramFuDA.setText(self.paramFuDAtext)#print(peakList)
#        #self.table.update(objectList=self.objectList, textMatrix=self.textMatrix)
#        #self.table.update()
#
#    def runFuDA(self):
#
#        #if self.proc_dir == self.spec_dir:
#        #    workdir = self.proc_dir
#        workdir = self.spec_dir
#        # make working dir if not already there
#        fudaOutDir = os.path.join(workdir,self.out_dir.get())
#        if os.path.exists(fudaOutDir):
#            message = "using pre-existing path %s as working dir"%fudaOutDir
#            print(message)
#        else:
#            message = "creating %s as working dir"%fudaOutDir
#            print(message)
#            os.system("mkdir %s"%fudaOutDir)
#        paramsFuDApath = os.path.join(fudaOutDir,"params.fuda")
#
#        if os.path.exists(paramsFuDApath):
#            os.system("cp %s %s"%(paramsFuDApath,paramsFuDApath+".old"))
#        # peaklist path    
#        peakListPath = os.path.join(fudaOutDir,"peaks.fuda")
#        # make new param file
#        paramFuDAfile = open(paramsFuDApath,"w")
#        paramFuDAfile.write(header.substitute(peaklist=peakListPath,
#            specfile=self.specFile.get(),noise=self.noiseEntry.get(),
#            zcoor=self.zcoorEntry.get(),delayfactor=self.delayEntry.get(),
#            verboselevel=self.verbEntry.get(),baseline=self.baseEntry.get(),
#            maxfev=self.maxfevEntry.get(),tol=self.tolEntry.get(),
#            fitexp=self.fitexpEntry.get(),printdata=self.printEntry.get(),
#            def_linewidth_f1=self.F1_lwEntry.get(),
#            def_linewidth_f2=self.F2_lwEntry.get(),
#            def_radius_f1=self.F1_radEntry.get(),
#            def_radius_f2=self.F2_radEntry.get(),
#            shape=self.shapeEntry.get(),
#            isotopeshift=self.isotopeshiftEntry.get()))
#        paramFuDAfile.write(self.paramFuDA.getText())
#        paramFuDAfile.close()
#        if os.path.exists(peakListPath):
#            os.system("cp %s %s"%(peakListPath,peakListPath+".old"))
#
#        peakListFile = open(peakListPath,"w")
#        peakListTemp = Template("$name\t$f1\t$f2\n")
#        peakListStr = peakListTemp.substitute(name=self.peakList[0].name,
#                f1=self.peakList[0].w1,f2=self.peakList[0].w2)
#
#        for num,i in enumerate(self.peakList[1:]):
#            if i.name in peakListStr:
#                # remove repeated peaks
#                self.peakList.pop(num+1)
#            else:
#                peakListStr+=peakListTemp.substitute(name=i.name,
#                        f1=i.w1,f2=i.w2)
#        peakListFile.write(peakListStr)
#        peakListFile.close()
#        outDir = os.path.join(fudaOutDir,"out")
#        os.system("runFuDA.sh %s %s"%(paramsFuDApath,outDir))
#        # add workdir path to specfile
#        #if workdir not in header_dic["specfile"]:
#        #    header_dic["specfile"] = os.path.join(workdir,header_dic["specfile"])
#        #else:
#        #    pass
#
#
#def radarSuperimposeMacro(argServer):
#  popup = RadarSuperimposePopup(argServer.parent)
#  popup.open()
#
#class RadarSuperimposePopup(BasePopup):
#
#  def __init__(self, parent, *args, **kw):
#    self.spectrum = None
#    self.dir = '.'
#
#    BasePopup.__init__(self, parent, *args, **kw)
#
#  def body(self, guiFrame):
#  
#    frame = LabelFrame(guiFrame, text='Options', grid=(0,0))
#    
#    label = Label(frame, text='Destination Spectrum:', grid=(0,0))
#    self.specPulldown = PulldownList(frame, callback=self.changeSpec, grid=(0,1))
#    
#    label = Label(frame, text='Peaklist File:', grid=(1,0))
#    self.peakListEntry = Entry(frame, text='', width=64, grid=(1,1))
#    button = Button(frame, command=self.importPeakFile,
#                    grid=(1,2), text='Choose File')
#    
#    label = Label(frame, text='Resonance file:', grid=(2,0))
#    self.resonanceFileEntry = Entry(frame, text='', width=64, grid=(2,1))
#    button = Button(frame, command=self.importResonanceFile,
#                    grid=(2,2), text='Choose File')
#    
#    texts = ['Make Peak List','Remove Peak List']
#    commands = [self.makePeakList, self.removePeakList]
#    buttons = UtilityButtonList(guiFrame, texts=texts,
#                                commands=commands, grid=(1,0))
#  
#    self.updateSpecPulldown()
#  
#    # Notifiers
#
#  def updateSpecPulldown(self):
#  
#    names = []
#    index = 0
#    spectra = []
#    
#    peakLists = getThroughSpacePeakLists(self.project)
#    for peakList in peakLists:
#      spectrum = peakList.dataSource
#      
#      if spectrum not in spectra:
#        names.append('%s:%s' % (spectrum.experiment.name, spectrum.name))
#        spectra.append(spectrum)
#    
#    if spectra:
#      if self.spectrum not in spectra:
#        self.spectrum = spectra[0]
#        
#      index = spectra.index(self.spectrum)
#    
#    else:
#      self.spectrum = None
#    
#    self.specPulldown.setup(names, spectra, index)
#
#
#
#  def changeSpec(self, spectrum):
#  
#    if spectrum is not self.spectrum:
#      self.spectrum = spectrum
#
#
#
#  def importResonanceFile(self):
#  
#    fileTypes = [ FileType('XEasy', ['*.resonances']), FileType('All', ['*']) ]
#    fileSelectPopup = FileSelectPopup(self, file_types=fileTypes, directory=self.dir,
#                        title='Import Xeasy Resonance file', dismiss_text='Cancel',
#                        selected_file_must_exist=True, multiSelect=False,)
#
#    file = fileSelectPopup.getFile()
#    
#    self.dir = fileSelectPopup.getDirectory()
#    self.resonanceFileEntry.set(file)
#
#
#
#  def importPeakFile(self):
#  
#    fileTypes = [  FileType('XEasy', ['*.peaks']), FileType('All', ['*'])]
#    fileSelectPopup = FileSelectPopup(self, file_types=fileTypes, directory=self.dir,
#                        title='Import XEasy peak file', dismiss_text='Cancel',
#                        selected_file_must_exist=True, multiSelect=False,)
#
#    file = fileSelectPopup.getFile()
#    
#    self.dir = fileSelectPopup.getDirectory()
#    self.peakListEntry.set(file)
#
#
#
#  def warning(self, msg, title='Failure'):
#  
#     showWarning(title, msg, parent=self)
#
#
#
#  def makePeakList(self):
#
#    if not self.spectrum:
#      self.warning('No spectrum')
#      return
#  
#    peakFile = self.peakListEntry.get()
#    resonanceFile = self.resonanceFileEntry.get()
#  
#    if not peakFile:
#      self.warning('No peak file specified')
#      return
#
#    if not resonanceFile:
#      self.warning('No resonance file specified')
#      return
#    
#    if not os.path.exists(peakFile):
#      self.warning('Specified peak file does not exist')
#      return
#    
#    if not os.path.exists(resonanceFile):
#      self.warning('Specified resonance file does not exist',)
#      return
#      
#    if not os.access(peakFile, READABLE):
#      self.warning('Specified peak file not readable')
#      return
#    
#    if not os.access(resonanceFile, READABLE):
#      self.warning('Specified resonance file not readable')
#      return
#
#
#    if peakList:
#      peaks = peakList.peaks
#
#      if peaks:
#        msg  = 'Destination peak list already contains %d peaks.' % (len(peaks))
#        msg += ' Remove these first?'
#        if showYesNo('Query', msg, parent=self):
#          for peak in peaks:
#            peak.delete()
#    
#    else:
#      peakList = self.spectrum.newPeakList(isSimulated=True,
#                                           name=TEMP_LIST_NAME)
#      analysisPeakList = getAnalysisPeakList(peakList)
#      analysisPeakList.symbolStyle = '+'
#      analysisPeakList.symbolColor = '#FF0000'
#      analysisPeakList.textColor = '#BB0000'
#
#
#
#  
#    resonanceDict = readResonanceData(resonanceFile)
#    nDim, dTypes, peakData = readPeakData(peakFile)
#
#
#
#
#    # Work out dim mapping
#    
#    xIsotope = None
#    xAtom = None
#    for atom, isotope in (('N', '15N'), ('C', '13C')):
#      if atom in dTypes:
#        xIsotope = isotope
#        xAtom = atom
#        
#    dataDims = self.spectrum.sortedDataDims()
#    boundDims = {}
#    for dataDim0, dataDim1 in getOnebondDataDims(self.spectrum):
#      boundDims[dataDim0] = True
#      boundDims[dataDim1] = True
#    
#    dimCols = []
#    if xAtom:
#      for dataDim in dataDims:
#        isotopes = getDataDimIsotopes(dataDim)
#        
#        if '1H' in isotopes:
#          if boundDims.get(dataDim):
#            col = dTypes.index('H'+xAtom)
#          else:
#            col = dTypes.index('H')
#            
#        elif xIsotope in isotopes:
#          col = dTypes.index(xAtom)
#          
#        dimCols.append(col)
#    
#    else:
#      # 2D NOESY - symmetric, can flip if wrong
#      dimCols = [0,1]  
#
#
#
#    # Write peaks
#    nDim = len(dataDims)
#    dims = range(nDim)
#    
#    c = 0
#    for num, ppms, inten, assign, dist in peakData:
#      
#      position = [None] * nDim
#      for i in dims:
#        position[i] = ppms[dimCols[i]]
#      
#      peak = pickPeak(peakList, position, unit='ppm')
#      c += 1
#      
#      labels = ['-'] * nDim
#      for i, peakDim in enumerate(peak.sortedPeakDims()):
#        j = assign[dimCols[i]]
#        
#        if j: # Not zero either
#          resonanceInfo = resonanceDict.get(j)
#          
#          if resonanceInfo:
#            resNum, atom, ppm, sd = resonanceInfo
#            labels[i] = '%d%s' % (resNum, atom)
#
#      peak.annotation = dist + ':' + ','.join(labels)
#     
#    showInfo('Done', 'Made %d peaks' % c, parent=self)
#
#
#
#  def removePeakList(self):
#
#    if self.spectrum:
#      peakList = self.spectrum.findFirstPeakList(name=TEMP_LIST_NAME,
#                                                 isSimulated=True)
#      
#      if not peakList:
#        self.warning('No temporary peak list found in this spectrum')
#        return
#        
#      name = '%s:%s' % (self.spectrum.name, self.spectrum.experiment.name)
#      msg = 'Really remove temporary peak list in spectrum %s?' % name
#      
#      if showOkCancel('Confirm', msg, parent=self):
#        peakList.delete()
#
#def readResonanceData(resonanceFile):
#  resonanceDict = {}
#  fileObj = open(resonanceFile, 'r')
#  line = fileObj.readline()
#  
#  while line:
#    array = line.split()
#    
#    if (len(array) == 5) and (array[0][0] != '#'):
#      num, ppm, sd, atom, resNum = array
#      resonanceDict[int(num)] = (int(resNum), atom, ppm, sd)
#    
#    line = fileObj.readline()
#  fileObj.close()
#  
#  return resonanceDict
#
#
#def readPeakData(peakFile):
#
#  nDim = 3
#  dTypes = [None] * 4
#  peakData = []
#  fileObj = open(peakFile, 'r')
#  line = fileObj.readline()
#  while line:
#    array = line.split()
#    
#    if array:
#      if array[0][0] == '#':
#        if 'dimensions' in line.lower():
#          nDim = int(array[-1])
#        
#        elif 'INAME' in line.upper():
#          dTypes[int(array[-2])-1] = array[-1]
#            
#      elif len(array) == 2*nDim+11:
#        num  = int(array[0])
#        ppms = [float(x) for x in array[1:nDim+1]]
#        inten = [float(x) for x in array[nDim+3:nDim+4]]
#        assign = [int(x) for x in array[nDim+7:nDim+nDim+7]]
#
#        dist = array[-1]
#        peakData.append((num, ppms, inten, assign, dist))
#        
#    line = fileObj.readline()
#  fileObj.close()
#  
#  while dTypes[-1] is None:
#    dTypes.pop()
#  
#  return nDim, dTypes, peakData
