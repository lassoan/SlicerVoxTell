import logging
import os
import random
import sys
import tempfile
import time

import qt
import slicer
import vtk
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# VoxTell
#


class VoxTell(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("VoxTell")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Andras Lasso (PerkLab, Queen's University)"]
        self.parent.helpText = _("""
3D Slicer extension for free-text promptable 3D medical image segmentation using VoxTell.
See more information in the <a href="https://github.com/lassoan/SlicerVoxTell">extension documentation</a>.
""")
        self.parent.acknowledgementText = _("""
This module uses <a href="https://github.com/MIC-DKFZ/VoxTell">VoxTell</a>,
a free-text promptable universal 3D medical image segmentation model developed at DKFZ.
""")


#
# VoxTellWidget
#


class VoxTellWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._didFirstEnterCheck = False

    def setup(self):
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/VoxTell.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = VoxTellLogic()

        # Connections
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputSegmentationChanged)
        self.ui.promptsTextEdit.connect("textChanged()", self.updateApplyButtonState)
        self.ui.promptsTextEdit.connect("textChanged()", self.onPromptsTextChanged)
        self.ui.deviceComboBox.connect("currentIndexChanged(int)", self.updateDeviceMemoryWarning)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.installDependenciesButton.connect("clicked(bool)", self.onInstallDependenciesButton)
        self.ui.modelPathBrowseButton.connect("clicked(bool)", self.onBrowseModelPath)
        self.ui.downloadModelButton.connect("clicked(bool)", self.onDownloadModel)

        self.initializeParameterNode()
        self.updateSetupStatus(collapseIfReady=True, collapseModelIfValid=True)
        self.updateApplyButtonState()
        self.updateDeviceMemoryWarning()

    def cleanup(self):
        """Called when the application closes and the module widget is destroyed."""
        if self.logic:
            self.logic.clearCaches()
        self.setParameterNode(None)
        self.removeObservers()

    def enter(self):
        """Called each time the user opens this module."""
        self.initializeParameterNode()
        firstEnter = not self._didFirstEnterCheck
        self.updateSetupStatus(collapseIfReady=firstEnter, collapseModelIfValid=firstEnter)
        self._didFirstEnterCheck = True

    def exit(self):
        """Called each time the user exits this module."""
        pass

    def onInputVolumeChanged(self, node):
        self.updateParameterNodeFromGUI()
        self.updateApplyButtonState()

    def onOutputSegmentationChanged(self, node):
        self.updateParameterNodeFromGUI()

    def onPromptsTextChanged(self):
        self.updateParameterNodeFromGUI()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.GetNodeReference("InputVolume"):
            self.selectDefaultInputVolume()
            inputNode = self.ui.inputVolumeSelector.currentNode()
            if inputNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", inputNode.GetID())

    def setParameterNode(self, inputParameterNode):
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onParameterNodeModified)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onParameterNodeModified)
        self.updateGUIFromParameterNode()

    def onParameterNodeModified(self, caller=None, event=None):
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self):
        if not self._parameterNode or self._updatingGUIFromParameterNode:
            return

        self._updatingGUIFromParameterNode = True
        try:
            inputVolumeNode = self._parameterNode.GetNodeReference("InputVolume")
            if self.ui.inputVolumeSelector.currentNode() != inputVolumeNode:
                self.ui.inputVolumeSelector.setCurrentNode(inputVolumeNode)

            outputSegmentationNode = self._parameterNode.GetNodeReference("OutputSegmentation")
            if self.ui.outputSegmentationSelector.currentNode() != outputSegmentationNode:
                self.ui.outputSegmentationSelector.setCurrentNode(outputSegmentationNode)

            promptsText = self._parameterNode.GetParameter("PromptsText")
            if promptsText is None:
                promptsText = ""
            if self.ui.promptsTextEdit.toPlainText() != promptsText:
                self.ui.promptsTextEdit.setPlainText(promptsText)
        finally:
            self._updatingGUIFromParameterNode = False

        self.updateApplyButtonState()

    def updateParameterNodeFromGUI(self):
        if not self._parameterNode or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()
        try:
            inputVolumeNode = self.ui.inputVolumeSelector.currentNode()
            self._parameterNode.SetNodeReferenceID("InputVolume", inputVolumeNode.GetID() if inputVolumeNode else None)

            outputSegmentationNode = self.ui.outputSegmentationSelector.currentNode()
            self._parameterNode.SetNodeReferenceID("OutputSegmentation", outputSegmentationNode.GetID() if outputSegmentationNode else None)

            self._parameterNode.SetParameter("PromptsText", self.ui.promptsTextEdit.toPlainText())
        finally:
            self._parameterNode.EndModify(wasModified)

    def selectDefaultInputVolume(self):
        if self.ui.inputVolumeSelector.currentNode():
            return
        firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        if firstVolumeNode:
            self.ui.inputVolumeSelector.setCurrentNode(firstVolumeNode)

    def appendStatusMessage(self, message):
        if not hasattr(self.ui, "segmentationStatusTextEdit"):
            return
        self.ui.segmentationStatusTextEdit.appendPlainText(message)
        cursor = self.ui.segmentationStatusTextEdit.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        self.ui.segmentationStatusTextEdit.setTextCursor(cursor)

    def updateApplyButtonState(self):
        inputVolume = self.ui.inputVolumeSelector.currentNode()
        prompts = self.ui.promptsTextEdit.toPlainText().strip()
        self.ui.applyButton.enabled = (inputVolume is not None and len(prompts) > 0)

    def updateDeviceMemoryWarning(self, _index=None):
        if not hasattr(self.ui, "statusLabel"):
            return

        warningText = ""
        useGpu = (self.ui.deviceComboBox.currentIndex == 0)
        if useGpu:
            try:
                import torch
                if torch.cuda.is_available():
                    totalMemoryBytes = torch.cuda.get_device_properties(0).total_memory
                    totalMemoryGb = totalMemoryBytes / (1024.0 ** 3)
                    if totalMemoryGb < 8.0:
                        warningText = _("Warning: Detected GPU memory is {0:.1f} GB. At least 8 GB is recommended.").format(totalMemoryGb)
            except Exception:
                warningText = ""

        self.ui.statusLabel.text = warningText
        self.ui.statusLabel.visible = bool(warningText)

    def updateSetupStatus(self, collapseIfReady=False, collapseModelIfValid=False):
        """Update the status labels in the Setup section."""
        dependenciesInstalled = self.logic.areDependenciesInstalled()
        incompatibleTorchVersion = self.logic.incompatibleTorchVersionString()
        if incompatibleTorchVersion:
            self.ui.dependenciesStatusLabel.text = _("Incompatible PyTorch installed ({0}). Please install a version other than 2.9.x.").format(incompatibleTorchVersion)
            self.ui.dependenciesStatusLabel.setStyleSheet("color: red")
        elif dependenciesInstalled:
            self.ui.dependenciesStatusLabel.text = _("Installed")
            self.ui.dependenciesStatusLabel.setStyleSheet("color: green")
        else:
            self.ui.dependenciesStatusLabel.text = _("Not installed")
            self.ui.dependenciesStatusLabel.setStyleSheet("color: red")

        modelPath = self.ui.modelPathLineEdit.text.strip() or self.logic.defaultModelPath()
        modelInstalled = self.logic.isModelInstalled(modelPath)
        if modelInstalled:
            self.ui.modelStatusLabel.text = _("Installed")
            self.ui.modelStatusLabel.setStyleSheet("color: green")
            if not self.ui.modelPathLineEdit.text.strip():
                self.ui.modelPathLineEdit.text = self.logic.defaultModelPath()
                modelPath = self.ui.modelPathLineEdit.text.strip()
        else:
            self.ui.modelStatusLabel.text = _("Not installed")
            self.ui.modelStatusLabel.setStyleSheet("color: red")

        if collapseIfReady and dependenciesInstalled and modelInstalled:
            self.ui.dependenciesCollapsibleButton.collapsed = True

        modelPathIsValidAndSet = bool(modelPath) and self.logic.isModelInstalled(modelPath)
        if collapseModelIfValid and modelPathIsValidAndSet:
            self.ui.modelCollapsibleButton.collapsed = True

    def onInstallDependenciesButton(self):
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.logic.setupPythonRequirements()
            if not self.ui.modelPathLineEdit.text.strip() and self.logic.isModelInstalled(self.logic.defaultModelPath()):
                self.ui.modelPathLineEdit.text = self.logic.defaultModelPath()
            self.updateSetupStatus()
            slicer.util.infoDisplay(_("Dependencies and model installed successfully."))
        except Exception as e:
            slicer.util.errorDisplay(_("Failed to install dependencies:\n") + str(e))
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onBrowseModelPath(self):
        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            _("Select VoxTell Model Directory"),
            self.ui.modelPathLineEdit.text or os.path.expanduser("~")
        )
        if path:
            self.ui.modelPathLineEdit.text = path

    def onDownloadModel(self):
        downloadDir = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            _("Select Directory to Download VoxTell Model"),
            self.ui.modelPathLineEdit.text or os.path.expanduser("~")
        )
        if not downloadDir:
            return

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            modelPath = self.logic.downloadModel(downloadDir)
            self.ui.modelPathLineEdit.text = modelPath
            self.updateSetupStatus()
            slicer.util.infoDisplay(_("Model downloaded successfully to:\n") + modelPath)
        except Exception as e:
            slicer.util.errorDisplay(_("Failed to download model:\n") + str(e))
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onApplyButton(self):
        inputVolume = self.ui.inputVolumeSelector.currentNode()
        if not inputVolume:
            slicer.util.errorDisplay(_("Please select an input volume."))
            return

        outputSegmentation = self.ui.outputSegmentationSelector.currentNode()

        promptsText = self.ui.promptsTextEdit.toPlainText().strip()
        if not promptsText:
            slicer.util.errorDisplay(_("Please enter at least one text prompt."))
            return

        prompts = [p.strip() for p in promptsText.splitlines() if p.strip()]

        modelPath = self.ui.modelPathLineEdit.text.strip()
        if not modelPath:
            slicer.util.errorDisplay(_("Please specify a model directory or download the model first."))
            return

        deviceIndex = self.ui.deviceComboBox.currentIndex
        useGpu = (deviceIndex == 0)
        deviceText = _("GPU (CUDA)") if useGpu else _("CPU")

        self.ui.segmentationStatusTextEdit.clear()
        self.appendStatusMessage(_("Starting segmentation..."))
        self.appendStatusMessage(_("Input volume: ") + inputVolume.GetName())
        self.appendStatusMessage(_("Device: ") + deviceText)
        self.appendStatusMessage(_("Prompts: ") + ", ".join(prompts))

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            segmentationNode = self.logic.runSegmentation(
                inputVolume,
                prompts,
                modelPath,
                useGpu,
                outputSegmentationNode=outputSegmentation,
                statusCallback=self.appendStatusMessage,
            )
            self.ui.outputSegmentationSelector.setCurrentNode(segmentationNode)
            self.appendStatusMessage(_("Segmentation completed successfully."))
            self.appendStatusMessage(_("Output segmentation: ") + segmentationNode.GetName())
        except Exception as e:
            self.appendStatusMessage(_("Segmentation failed: ") + str(e))
            slicer.util.errorDisplay(_("Segmentation failed:\n") + str(e))
            import traceback
            logging.error(traceback.format_exc())
        finally:
            qt.QApplication.restoreOverrideCursor()


#
# VoxTellLogic
#


class VoxTellLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual computation done by your module.
    The interface should be such that other Python code can import this class and
    make use of the functionality without requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    MODEL_NAME = "voxtell_v1.1"
    DEFAULT_TERMINOLOGY_CONTEXT = "Segmentation category and type - 3D Slicer General Anatomy list"
    TORCH_VERSION_REQUIREMENT = "!=2.9.*"  # avoid PyTorch 2.9.x due to known compatibility issues with VoxTell as of Feb 2026
    FAST_INFERENCE_STEP_SIZE = 0.75

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._cachedPredictor = None
        self._cachedPredictorModelPath = None
        self._cachedPredictorDevice = None
        self._cachedInputVolumeSignature = None
        self._cachedPreprocessedData = None
        self._cachedPreprocessedBbox = None
        self._cachedPreprocessedOrigShape = None
        self._cachedImageProperties = None
        self._cachedTextEmbeddings = {}
        self._maxCachedTextEmbeddings = 256

    def clearCaches(self):
        self._cachedPredictor = None
        self._cachedPredictorModelPath = None
        self._cachedPredictorDevice = None
        self._cachedInputVolumeSignature = None
        self._cachedPreprocessedData = None
        self._cachedPreprocessedBbox = None
        self._cachedPreprocessedOrigShape = None
        self._cachedImageProperties = None
        self._cachedTextEmbeddings.clear()
        logging.info("Cleared VoxTell logic caches.")

    def _getOrCreatePredictor(self, modelPath, device, statusCallback=None):
        from voxtell.inference.predictor import VoxTellPredictor

        requestedDevice = str(device)
        if (
            self._cachedPredictor is not None
            and self._cachedPredictorModelPath == modelPath
            and self._cachedPredictorDevice == requestedDevice
        ):
            logging.info("Reusing cached VoxTell predictor (%s, %s).", modelPath, requestedDevice)
            if statusCallback:
                statusCallback(_("Reusing already initialized model."))
            return self._cachedPredictor

        if statusCallback:
            statusCallback(_("Loading model..."))
        loadStartTime = time.perf_counter()
        predictor = VoxTellPredictor(
            model_dir=modelPath,
            device=device,
        )
        modelLoadSec = time.perf_counter() - loadStartTime
        logging.info("Loaded VoxTell predictor in %.2f s (%s, %s).", modelLoadSec, modelPath, requestedDevice)
        if statusCallback:
            statusCallback(_("Model loaded in {0:.2f} s.").format(modelLoadSec))
        self._cachedPredictor = predictor
        self._cachedPredictorModelPath = modelPath
        self._cachedPredictorDevice = requestedDevice
        return predictor

    def _inputVolumeSignature(self, inputVolumeNode):
        if inputVolumeNode is None:
            return None

        imageData = inputVolumeNode.GetImageData()
        imageDataMTime = imageData.GetMTime() if imageData else None
        dimensions = imageData.GetDimensions() if imageData else None
        return (
            inputVolumeNode.GetID(),
            imageDataMTime,
            dimensions,
        )

    def _getOrCreatePreprocessedInput(self, inputVolumeNode, predictor, statusCallback=None):
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        currentSignature = self._inputVolumeSignature(inputVolumeNode)
        if (
            self._cachedPreprocessedData is not None
            and self._cachedPreprocessedBbox is not None
            and self._cachedPreprocessedOrigShape is not None
            and self._cachedImageProperties is not None
            and self._cachedInputVolumeSignature == currentSignature
        ):
            logging.info("Reusing cached preprocessed input volume.")
            if statusCallback:
                statusCallback(_("Reusing preprocessed input volume."))
            return (
                self._cachedPreprocessedData,
                self._cachedPreprocessedBbox,
                self._cachedPreprocessedOrigShape,
                self._cachedImageProperties,
            )

        if statusCallback:
            statusCallback(_("Preparing input volume..."))

        imageIO = NibabelIOWithReorient()
        preprocessStartTime = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpDir:
            inputNiftiPath = os.path.join(tmpDir, "input.nii.gz")
            slicer.util.exportNode(inputVolumeNode, inputNiftiPath)
            img, imageProperties = imageIO.read_images([inputNiftiPath])
            preprocessedData, preprocessedBbox, preprocessedOrigShape = predictor.preprocess(img)
        preprocessSec = time.perf_counter() - preprocessStartTime
        logging.info(
            "Prepared input volume in %.2f s (shape=%s, bbox=%s).",
            preprocessSec,
            getattr(preprocessedData, "shape", None),
            preprocessedBbox,
        )
        if statusCallback:
            statusCallback(_("Input preprocessing completed in {0:.2f} s.").format(preprocessSec))

        self._cachedPreprocessedData = preprocessedData
        self._cachedPreprocessedBbox = preprocessedBbox
        self._cachedPreprocessedOrigShape = preprocessedOrigShape
        self._cachedImageProperties = imageProperties
        self._cachedInputVolumeSignature = currentSignature
        return preprocessedData, preprocessedBbox, preprocessedOrigShape, imageProperties

    def _cacheTextEmbedding(self, modelPath, device, prompt, embeddingCpu):
        key = (modelPath, str(device), prompt)
        self._cachedTextEmbeddings[key] = embeddingCpu
        if len(self._cachedTextEmbeddings) > self._maxCachedTextEmbeddings:
            oldestKey = next(iter(self._cachedTextEmbeddings))
            del self._cachedTextEmbeddings[oldestKey]

    def _getOrCreateTextEmbeddings(self, predictor, modelPath, device, textPrompts, statusCallback=None):
        import torch

        embeddingsToConcatenate = []
        missingPrompts = []
        missingPromptIndices = []
        cacheHits = 0

        for index, prompt in enumerate(textPrompts):
            key = (modelPath, str(device), prompt)
            cachedEmbedding = self._cachedTextEmbeddings.get(key)
            if cachedEmbedding is not None:
                cacheHits += 1
                embeddingsToConcatenate.append(cachedEmbedding)
            else:
                missingPrompts.append(prompt)
                missingPromptIndices.append(index)
                embeddingsToConcatenate.append(None)

        embedSec = 0.0
        if missingPrompts:
            if statusCallback:
                statusCallback(_("Embedding {0} new prompt(s)...").format(len(missingPrompts)))
            embedStartTime = time.perf_counter()
            missingEmbeddings = predictor.embed_text_prompts(missingPrompts)
            embedSec = time.perf_counter() - embedStartTime
            missingEmbeddingsCpu = missingEmbeddings.detach().to("cpu")

            for i, prompt in enumerate(missingPrompts):
                embeddingCpu = missingEmbeddingsCpu[i:i + 1].clone()
                self._cacheTextEmbedding(modelPath, device, prompt, embeddingCpu)
                targetIndex = missingPromptIndices[i]
                embeddingsToConcatenate[targetIndex] = embeddingCpu

            logging.info("Embedded %d new prompt(s) in %.2f s.", len(missingPrompts), embedSec)

        assembleStartTime = time.perf_counter()
        textEmbeddings = torch.cat(embeddingsToConcatenate, dim=0).to(device)
        assembleSec = time.perf_counter() - assembleStartTime
        logging.info(
            "Prepared %d prompt embedding(s): %d cache hit(s), %d new; assembly/transfer %.2f s.",
            len(textPrompts),
            cacheHits,
            len(missingPrompts),
            assembleSec,
        )

        return textEmbeddings, cacheHits, len(missingPrompts), embedSec, assembleSec

    def _buildFastInferenceKwargs(self, predictor):
        import inspect

        method = predictor.predict_sliding_window_return_logits
        parameters = inspect.signature(method).parameters
        kwargs = {}

        if "step_size" in parameters:
            kwargs["step_size"] = self.FAST_INFERENCE_STEP_SIZE
        if "overlap" in parameters:
            kwargs["overlap"] = 1.0 - self.FAST_INFERENCE_STEP_SIZE
        if "use_gaussian" in parameters:
            kwargs["use_gaussian"] = False
        if "do_tta" in parameters:
            kwargs["do_tta"] = False
        if "use_mirroring" in parameters:
            kwargs["use_mirroring"] = False

        return kwargs

    def _predictSlidingWindowLogits(self, predictor, preprocessedData, textEmbeddings, statusCallback=None):
        inferenceKwargs = self._buildFastInferenceKwargs(predictor)
        if inferenceKwargs:
            logging.info("Running fast sliding-window inference kwargs: %s", str(inferenceKwargs))
            if statusCallback:
                statusCallback(_("Using fast sliding-window settings."))
        return predictor.predict_sliding_window_return_logits(preprocessedData, textEmbeddings, **inferenceKwargs)

    def _randomSegmentColor(self):
        return (random.random(), random.random(), random.random())

    def _terminologiesLogic(self):
        terminologiesModule = getattr(slicer.modules, "terminologies", None)
        if not terminologiesModule:
            return None
        return terminologiesModule.logic()

    def _terminologyContextsToTry(self, terminologiesLogic):
        contexts = [self.DEFAULT_TERMINOLOGY_CONTEXT]
        try:
            import vtk
            loadedTerminologyNames = vtk.vtkStringArray()
            terminologiesLogic.GetLoadedTerminologyNames(loadedTerminologyNames)
            for i in range(loadedTerminologyNames.GetNumberOfValues()):
                contextName = loadedTerminologyNames.GetValue(i)
                if contextName and contextName not in contexts:
                    contexts.append(contextName)
        except Exception:
            pass
        return contexts

    def _findTerminologyAndColor(self, segmentName):
        terminologiesLogic = self._terminologiesLogic()
        if not terminologiesLogic:
            return None, None

        entry = slicer.vtkSlicerTerminologyEntry()
        for contextName in self._terminologyContextsToTry(terminologiesLogic):
            try:
                found = terminologiesLogic.FindTypeInTerminologyBy3dSlicerLabel(contextName, segmentName, entry)
            except Exception:
                continue
            if not found:
                continue

            terminologyEntrySerialized = terminologiesLogic.SerializeTerminologyEntry(entry)
            if not terminologyEntrySerialized:
                continue

            typeObject = entry.GetTypeObject()
            if typeObject and typeObject.GetHasModifiers() and entry.GetTypeModifierObject() and entry.GetTypeModifierObject().GetCodeValue():
                colorRgb = entry.GetTypeModifierObject().GetRecommendedDisplayRGBValue()
            elif typeObject:
                colorRgb = typeObject.GetRecommendedDisplayRGBValue()
            else:
                colorRgb = None

            if colorRgb and list(colorRgb) != [127, 127, 127]:
                color = tuple(component / 255.0 for component in colorRgb)
            else:
                color = None

            return terminologyEntrySerialized, color

        return None, None

    def _setSegmentTerminologyAndColor(self, segment, segmentName):
        terminologyEntrySerialized, terminologyColor = self._findTerminologyAndColor(segmentName)
        if terminologyEntrySerialized:
            segment.SetTerminology(terminologyEntrySerialized)
            if terminologyColor:
                segment.SetColor(*terminologyColor)
                return "terminology"

        segment.SetColor(*self._randomSegmentColor())
        return "random"

    def defaultModelPath(self):
        """Return the default path where the VoxTell model is stored."""
        userDataDirectory = qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation)
        if not userDataDirectory:
            userDataDirectory = os.path.expanduser("~")
        return os.path.join(userDataDirectory, "voxtell", self.MODEL_NAME)

    def areDependenciesInstalled(self):
        """Check if the required Python packages are installed."""
        import importlib.util
        return (
            importlib.util.find_spec("voxtell") is not None
            and importlib.util.find_spec("huggingface_hub") is not None
            and self.torchVersionIsSupported()
        )

    def _isTorchVersionAllowed(self, versionCore):
        requirement = self.TORCH_VERSION_REQUIREMENT.strip()
        if requirement.startswith("!=") and requirement.endswith(".*"):
            disallowedPrefix = requirement[2:-2]
            return not (versionCore == disallowedPrefix or versionCore.startswith(disallowedPrefix + "."))
        return True

    def _disallowedTorchVersionLabel(self):
        requirement = self.TORCH_VERSION_REQUIREMENT.strip()
        if requirement.startswith("!=") and requirement.endswith(".*"):
            return requirement[2:-2] + ".x"
        return requirement

    def installedTorchVersionString(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return None
        try:
            import torch
            return str(torch.__version__).split("+")[0]
        except Exception:
            return None

    def torchVersionIsSupported(self):
        versionCore = self.installedTorchVersionString()
        if versionCore is None:
            return False
        return self._isTorchVersionAllowed(versionCore)

    def incompatibleTorchVersionString(self):
        versionCore = self.installedTorchVersionString()
        if versionCore is None:
            return None
        if not self._isTorchVersionAllowed(versionCore):
            return versionCore
        return None

    def isModelInstalled(self, modelPath=None):
        """Check if the VoxTell model is installed at the given path."""
        if modelPath is None:
            modelPath = self.defaultModelPath()
        return bool(modelPath) and os.path.isdir(modelPath) and len(os.listdir(modelPath)) > 0

    def installPyTorchWithPyTorchUtils(self):
        """Install PyTorch using the PyTorchUtils extension, enforcing TORCH_VERSION_REQUIREMENT."""
        import inspect

        if self.torchVersionIsSupported():
            logging.info(f"torch is already installed and satisfies torch{self.TORCH_VERSION_REQUIREMENT}.")
            return

        pytorchUtilsLogic = None
        import importlib.util
        if importlib.util.find_spec("PyTorchUtils") is not None:
            import PyTorchUtils
            pytorchUtilsLogic = PyTorchUtils.PyTorchUtilsLogic()

        if pytorchUtilsLogic is None:
            pytorchUtilsModule = getattr(slicer.modules, "pytorchutils", None)
            if pytorchUtilsModule and hasattr(pytorchUtilsModule, "logic"):
                pytorchUtilsLogic = pytorchUtilsModule.logic()

        if pytorchUtilsLogic is None:
            raise RuntimeError(
                "PyTorchUtils extension is required to install PyTorch. "
                "Please install the 'PyTorch' extension from the Extension Manager, then retry."
            )

        installTorchMethod = getattr(pytorchUtilsLogic, "installTorch", None)
        if callable(installTorchMethod):
            logging.info(f"Installing torch using PyTorchUtils extension with requirement torch{self.TORCH_VERSION_REQUIREMENT}...")
            installSignature = inspect.signature(installTorchMethod)
            installParameters = installSignature.parameters
            installKwargs = {}
            if "askConfirmation" in installParameters:
                installKwargs["askConfirmation"] = False
            if "torchVersionRequirement" in installParameters:
                installKwargs["torchVersionRequirement"] = self.TORCH_VERSION_REQUIREMENT
            installTorchMethod(**installKwargs)
        else:
            logging.info("Installing torch using PyTorchUtils module logic...")
            pytorchUtilsLogic.torch

        if not self.torchVersionIsSupported():
            try:
                import torch
                installedVersion = str(torch.__version__)
            except Exception:
                installedVersion = "unknown"
            raise RuntimeError(
                f"Installed torch version ({installedVersion}) does not satisfy required version constraint torch{self.TORCH_VERSION_REQUIREMENT}."
            )

    def setupPythonRequirements(self):
        """Install required Python packages and download the AI model if not already installed."""
        import importlib.util

        self.installPyTorchWithPyTorchUtils()

        # Install voxtell (this will also install nnunetv2 and other dependencies)
        if importlib.util.find_spec("voxtell") is None:
            logging.info("Installing voxtell...")
            slicer.util.pip_install("voxtell")
        else:
            logging.info("voxtell is already installed.")

        # Install huggingface_hub if not present (for model downloading)
        if importlib.util.find_spec("huggingface_hub") is None:
            logging.info("Installing huggingface_hub...")
            slicer.util.pip_install("huggingface_hub")
        else:
            logging.info("huggingface_hub is already installed.")

        # Download model to default location if not already installed
        if not self.isModelInstalled():
            modelPath = self.defaultModelPath()
            os.makedirs(os.path.dirname(modelPath), exist_ok=True)
            self.downloadModel(os.path.dirname(modelPath))

    def downloadModel(self, downloadDir, modelName=None):
        """Download VoxTell model weights from Hugging Face.

        :param downloadDir: Directory where the model will be downloaded.
        :param modelName: Name of the model version to download (e.g., "voxtell_v1.1").
            See https://huggingface.co/mrokuss/VoxTell for available versions.
        :return: Path to the downloaded model directory.
        """
        if modelName is None:
            modelName = self.MODEL_NAME
        import importlib.util
        if importlib.util.find_spec("huggingface_hub") is None:
            slicer.util.pip_install("huggingface_hub")

        from huggingface_hub import snapshot_download
        logging.info(f"Downloading VoxTell model '{modelName}' to '{downloadDir}'...")
        downloadPath = snapshot_download(
            repo_id="mrokuss/VoxTell",
            allow_patterns=[f"{modelName}/*", "*.json"],
            local_dir=downloadDir,
        )
        modelPath = os.path.join(downloadPath, modelName)
        logging.info(f"Model downloaded to: {modelPath}")
        return modelPath

    def runSegmentation(self, inputVolumeNode, textPrompts, modelPath, useGpu=True, outputSegmentationNode=None, statusCallback=None):
        """Run VoxTell segmentation.

        :param inputVolumeNode: Input vtkMRMLScalarVolumeNode.
        :param textPrompts: List of text prompts (strings).
        :param modelPath: Path to the VoxTell model directory.
        :param useGpu: Whether to use GPU for inference.
        :param outputSegmentationNode: Optional existing vtkMRMLSegmentationNode to update.
        :param statusCallback: Optional callable that receives status text updates.
        :return: The created vtkMRMLSegmentationNode.
        """
        import torch
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        import numpy as np
        segmentationStartTime = time.perf_counter()

        # Select device
        if useGpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        if device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

        logging.info(f"Using device: {device}")
        if statusCallback:
            statusCallback(_("Using device: ") + str(device))

        predictor = self._getOrCreatePredictor(modelPath, device, statusCallback=statusCallback)
        preprocessedData, preprocessedBbox, preprocessedOrigShape, imageProperties = self._getOrCreatePreprocessedInput(
            inputVolumeNode,
            predictor,
            statusCallback=statusCallback,
        )
        imageIO = NibabelIOWithReorient()

        # Use a temporary NIfTI directory for generated masks
        with tempfile.TemporaryDirectory() as tmpDir:
            # Run prediction using lower-level API to reuse preprocessed input
            logging.info(f"Running VoxTell prediction with prompts: {textPrompts}")
            if statusCallback:
                statusCallback(_("Embedding prompts..."))
            textEmbeddings, embeddingCacheHits, embeddedPromptCount, embedSec, embeddingAssembleSec = self._getOrCreateTextEmbeddings(
                predictor,
                modelPath,
                device,
                textPrompts,
                statusCallback=statusCallback,
            )
            if statusCallback:
                statusCallback(
                    _("Prompt embeddings ready in {0:.2f} s ({1} cached, {2} new, transfer {3:.2f} s).")
                    .format(embedSec, embeddingCacheHits, embeddedPromptCount, embeddingAssembleSec)
                )

            if statusCallback:
                statusCallback(_("Running sliding-window prediction..."))
            inferenceStartTime = time.perf_counter()
            ampEnabled = (device.type == "cuda")
            if ampEnabled and statusCallback:
                statusCallback(_("Using CUDA mixed precision for inference."))
            try:
                with torch.inference_mode():
                    if ampEnabled:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            prediction = self._predictSlidingWindowLogits(
                                predictor,
                                preprocessedData,
                                textEmbeddings,
                                statusCallback=statusCallback,
                            )
                    else:
                        prediction = self._predictSlidingWindowLogits(
                            predictor,
                            preprocessedData,
                            textEmbeddings,
                            statusCallback=statusCallback,
                        )
            except Exception as inferenceException:
                if ampEnabled:
                    logging.warning(
                        "Mixed-precision inference failed, retrying in full precision. Error: %s",
                        str(inferenceException),
                    )
                    if statusCallback:
                        statusCallback(_("Mixed precision failed, retrying full-precision inference."))
                    with torch.inference_mode():
                        prediction = self._predictSlidingWindowLogits(
                            predictor,
                            preprocessedData,
                            textEmbeddings,
                            statusCallback=statusCallback,
                        )
                else:
                    raise
            inferenceSec = time.perf_counter() - inferenceStartTime
            logging.info("Sliding-window inference completed in %.2f s (device=%s).", inferenceSec, str(device))
            if statusCallback:
                statusCallback(_("Sliding-window inference completed in {0:.2f} s.").format(inferenceSec))

            transferStartTime = time.perf_counter()
            prediction = prediction.to("cpu")
            transferSec = time.perf_counter() - transferStartTime
            logging.info("Transferred logits to CPU in %.2f s.", transferSec)
            if statusCallback:
                statusCallback(_("Transfer to CPU completed in {0:.2f} s.").format(transferSec))

            thresholdStartTime = time.perf_counter()
            with torch.no_grad():
                prediction = torch.sigmoid(prediction.float()) > 0.5
            thresholdSec = time.perf_counter() - thresholdStartTime
            logging.info("Applied sigmoid/threshold in %.2f s.", thresholdSec)
            if statusCallback:
                statusCallback(_("Sigmoid and threshold completed in {0:.2f} s.").format(thresholdSec))

            cropInsertStartTime = time.perf_counter()
            voxtellSeg = np.zeros([prediction.shape[0], *preprocessedOrigShape], dtype=np.uint8)
            voxtellSeg = insert_crop_into_image(voxtellSeg, prediction, preprocessedBbox)
            cropInsertSec = time.perf_counter() - cropInsertStartTime
            logging.info("Inserted prediction crop into full image in %.2f s.", cropInsertSec)
            if statusCallback:
                statusCallback(_("Post-processing crop insertion completed in {0:.2f} s.").format(cropInsertSec))
            if statusCallback:
                statusCallback(_("Prediction completed."))

            if outputSegmentationNode is not None:
                segmentationNode = outputSegmentationNode
                clearStartTime = time.perf_counter()
                segmentationNode.GetSegmentation().RemoveAllSegments()
                clearSec = time.perf_counter() - clearStartTime
                logging.info("Cleared existing segmentation in %.2f s.", clearSec)
                if statusCallback:
                    statusCallback(_("Updating existing segmentation: ") + segmentationNode.GetName())
            else:
                # Create a segmentation node in Slicer
                segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                segmentationNode.SetName(inputVolumeNode.GetName() + "_VoxTell")
                if statusCallback:
                    statusCallback(_("Created new segmentation: ") + segmentationNode.GetName())

            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolumeNode)
            segmentationNode.CreateDefaultDisplayNodes()
            segmentation = segmentationNode.GetSegmentation()
            closedSurfaceRepresentationName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
            if segmentation.ContainsRepresentation(closedSurfaceRepresentationName):
                removeClosedSurfaceStartTime = time.perf_counter()
                segmentation.RemoveRepresentation(closedSurfaceRepresentationName)
                removeClosedSurfaceSec = time.perf_counter() - removeClosedSurfaceStartTime
                logging.info("Removed existing closed-surface representation in %.2f s.", removeClosedSurfaceSec)
                if statusCallback:
                    statusCallback(_("Temporarily removed closed-surface representation to speed segment updates."))

            # Add each prompt result as a segment
            for i, prompt in enumerate(textPrompts):
                segmentStartTime = time.perf_counter()
                if statusCallback:
                    statusCallback(_("Creating segment: ") + prompt)
                maskArray = voxtellSeg[i].astype(np.uint8)

                # Restore mask from the reoriented RAS space back to the original image orientation
                # before loading into Slicer to avoid RAS/LPS flips.
                maskNiftiPath = os.path.join(tmpDir, f"mask_{i}.nii.gz")
                imageIO.write_seg(maskArray, maskNiftiPath, imageProperties)

                labelVolumeNode = slicer.util.loadLabelVolume(maskNiftiPath)
                if labelVolumeNode is None:
                    raise RuntimeError(f"Failed to load generated label volume: {maskNiftiPath}")

                try:
                    # Import label map into segmentation
                    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                        labelVolumeNode, segmentationNode
                    )

                    # Rename the last added segment to the prompt text
                    lastSegmentId = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
                    segment = segmentation.GetSegment(lastSegmentId)
                    segment.SetName(prompt)
                    colorSource = self._setSegmentTerminologyAndColor(segment, prompt)
                    if statusCallback:
                        if colorSource == "terminology":
                            statusCallback(_("Assigned DICOM terminology and terminology-based color: ") + prompt)
                        else:
                            statusCallback(_("No matching terminology found, assigned random color: ") + prompt)
                finally:
                    # Remove the temporary label map node
                    slicer.mrmlScene.RemoveNode(labelVolumeNode)
                segmentSec = time.perf_counter() - segmentStartTime
                logging.info("Segment %d/%d ('%s') created in %.2f s.", i + 1, len(textPrompts), prompt, segmentSec)
                if statusCallback:
                    statusCallback(_("Segment '{0}' completed in {1:.2f} s.").format(prompt, segmentSec))

        # Show segmentation in 3D
        segmentationNode.CreateClosedSurfaceRepresentation()
        totalSegmentationSec = time.perf_counter() - segmentationStartTime
        logging.info("VoxTell segmentation pipeline completed in %.2f s.", totalSegmentationSec)
        if statusCallback:
            statusCallback(_("Total segmentation pipeline completed in {0:.2f} s.").format(totalSegmentationSec))
        if statusCallback:
            statusCallback(_("3D surface representation created."))

        return segmentationNode


#
# VoxTellTest
#


class VoxTellTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_VoxTellInstantiation()
        self.test_VoxTellLogicHelpers()
        self.test_VoxTellSegmentationIfReady()

    def test_VoxTellInstantiation(self):
        """Verify that the module can be instantiated and logic created."""
        self.delayDisplay("Testing VoxTell module instantiation")
        logic = VoxTellLogic()
        self.assertIsNotNone(logic)
        self.delayDisplay("VoxTell module instantiation test passed.")

    def test_VoxTellLogicHelpers(self):
        """Verify helper methods that do not require model inference."""
        self.delayDisplay("Testing VoxTell logic helper methods")
        logic = VoxTellLogic()

        defaultModelPath = logic.defaultModelPath()
        self.assertTrue(isinstance(defaultModelPath, str) and len(defaultModelPath) > 0)

        # Non-existent directory should not be considered a valid installed model.
        nonExistingModelPath = os.path.join(slicer.app.temporaryPath, "VoxTellModelThatDoesNotExist")
        self.assertFalse(logic.isModelInstalled(nonExistingModelPath))

        dependenciesInstalled = logic.areDependenciesInstalled()
        self.assertTrue(isinstance(dependenciesInstalled, bool))
        self.delayDisplay("VoxTell logic helper methods test passed.")

    def test_VoxTellSegmentationIfReady(self):
        """Run segmentation test only if dependencies and model are already available.

        If requirements are not met, this test exits successfully without running inference.
        """
        self.delayDisplay("Checking VoxTell runtime readiness for segmentation test")
        logic = VoxTellLogic()

        if not logic.areDependenciesInstalled():
            self.delayDisplay("Skipping segmentation test: dependencies are not installed.")
            return

        modelPath = logic.defaultModelPath()
        if not logic.isModelInstalled(modelPath):
            self.delayDisplay("Skipping segmentation test: model is not downloaded.")
            return

        self.delayDisplay("Dependencies and model found. Running CTACardio segmentation test.")
        import SampleData

        inputVolumeNode = SampleData.downloadSample("CTACardio")
        self.assertIsNotNone(inputVolumeNode)

        prompts = ["left ventricle", "aorta", "ribs", "vertebrae", "liver"]
        useGpu = False
        try:
            import torch
            useGpu = torch.cuda.is_available()
        except Exception:
            useGpu = False

        outputSegmentationNode = logic.runSegmentation(
            inputVolumeNode=inputVolumeNode,
            textPrompts=prompts,
            modelPath=modelPath,
            useGpu=useGpu,
            outputSegmentationNode=None,
            statusCallback=None,
        )

        self.assertIsNotNone(outputSegmentationNode)
        segmentation = outputSegmentationNode.GetSegmentation()
        self.assertEqual(segmentation.GetNumberOfSegments(), len(prompts))
        self.delayDisplay("CTACardio segmentation test passed.")
