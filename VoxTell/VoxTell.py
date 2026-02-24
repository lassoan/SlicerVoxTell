import logging
import os
import sys
import tempfile

import vtk
import qt
import slicer
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
        self.ui.promptsTextEdit.connect("textChanged()", self.updateApplyButtonState)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.installDependenciesButton.connect("clicked(bool)", self.onInstallDependenciesButton)
        self.ui.modelPathBrowseButton.connect("clicked(bool)", self.onBrowseModelPath)
        self.ui.downloadModelButton.connect("clicked(bool)", self.onDownloadModel)

        self.updateSetupStatus()
        self.updateApplyButtonState()

    def cleanup(self):
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self):
        """Called each time the user opens this module."""
        pass

    def exit(self):
        """Called each time the user exits this module."""
        pass

    def onInputVolumeChanged(self, node):
        self.updateApplyButtonState()

    def updateApplyButtonState(self):
        inputVolume = self.ui.inputVolumeSelector.currentNode()
        prompts = self.ui.promptsTextEdit.toPlainText().strip()
        self.ui.applyButton.enabled = (inputVolume is not None and len(prompts) > 0)

    def updateSetupStatus(self):
        """Update the status labels in the Setup section."""
        if self.logic.areDependenciesInstalled():
            self.ui.dependenciesStatusLabel.text = _("Installed")
            self.ui.dependenciesStatusLabel.setStyleSheet("color: green")
        else:
            self.ui.dependenciesStatusLabel.text = _("Not installed")
            self.ui.dependenciesStatusLabel.setStyleSheet("color: red")

        modelPath = self.ui.modelPathLineEdit.text.strip() or self.logic.defaultModelPath()
        if self.logic.isModelInstalled(modelPath):
            self.ui.modelStatusLabel.text = _("Installed")
            self.ui.modelStatusLabel.setStyleSheet("color: green")
            if not self.ui.modelPathLineEdit.text.strip():
                self.ui.modelPathLineEdit.text = self.logic.defaultModelPath()
        else:
            self.ui.modelStatusLabel.text = _("Not installed")
            self.ui.modelStatusLabel.setStyleSheet("color: red")

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

        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            segmentationNode = self.logic.runSegmentation(inputVolume, prompts, modelPath, useGpu)
            slicer.util.infoDisplay(_("Segmentation completed successfully."))
        except Exception as e:
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

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def defaultModelPath(self):
        """Return the default path where the VoxTell model is stored."""
        if hasattr(slicer.app, "userDataDirectory"):
            userDataDir = slicer.app.userDataDirectory()
        else:
            userDataDir = qt.QStandardPaths.writableLocation(qt.QStandardPaths.AppDataLocation)
        return os.path.join(userDataDir, "voxtell", self.MODEL_NAME)

    def areDependenciesInstalled(self):
        """Check if the required Python packages are installed."""
        import importlib.util
        return importlib.util.find_spec("voxtell") is not None

    def isModelInstalled(self, modelPath=None):
        """Check if the VoxTell model is installed at the given path."""
        if modelPath is None:
            modelPath = self.defaultModelPath()
        return bool(modelPath) and os.path.isdir(modelPath) and len(os.listdir(modelPath)) > 0

    def setupPythonRequirements(self):
        """Install required Python packages and download the AI model if not already installed."""
        import importlib.util

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

    def runSegmentation(self, inputVolumeNode, textPrompts, modelPath, useGpu=True):
        """Run VoxTell segmentation.

        :param inputVolumeNode: Input vtkMRMLScalarVolumeNode.
        :param textPrompts: List of text prompts (strings).
        :param modelPath: Path to the VoxTell model directory.
        :param useGpu: Whether to use GPU for inference.
        :return: The created vtkMRMLSegmentationNode.
        """
        import torch
        from voxtell.inference.predictor import VoxTellPredictor
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        import numpy as np

        # Select device
        if useGpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        logging.info(f"Using device: {device}")

        # Export the volume to a temporary NIfTI file
        with tempfile.TemporaryDirectory() as tmpDir:
            inputNiftiPath = os.path.join(tmpDir, "input.nii.gz")
            slicer.util.exportNode(inputVolumeNode, inputNiftiPath)

            # Load image using NibabelIOWithReorient, which automatically reorients
            # the image to RAS orientation as required by VoxTell for correct
            # anatomical localization (e.g., distinguishing left from right).
            img, _ = NibabelIOWithReorient().read_images([inputNiftiPath])

            # Initialize predictor
            predictor = VoxTellPredictor(
                model_dir=modelPath,
                device=device,
            )

            # Run prediction - output shape: (num_prompts, x, y, z)
            logging.info(f"Running VoxTell prediction with prompts: {textPrompts}")
            voxtellSeg = predictor.predict_single_image(img, textPrompts)

        # Create a segmentation node in Slicer
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(inputVolumeNode.GetName() + "_VoxTell")
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolumeNode)
        segmentationNode.CreateDefaultDisplayNodes()

        # Add each prompt result as a segment
        for i, prompt in enumerate(textPrompts):
            maskArray = voxtellSeg[i].astype(np.uint8)

            # Create a label map volume node from the mask
            labelVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            labelVolumeNode.SetName(f"tmp_{prompt}")

            # Copy geometry from input volume
            labelVolumeNode.CopyOrientation(inputVolumeNode)
            labelVolumeNode.SetSpacing(inputVolumeNode.GetSpacing())
            labelVolumeNode.SetOrigin(inputVolumeNode.GetOrigin())
            ijkToRASDirections = vtk.vtkMatrix4x4()
            inputVolumeNode.GetIJKToRASDirectionMatrix(ijkToRASDirections)
            labelVolumeNode.SetIJKToRASDirectionMatrix(ijkToRASDirections)

            # Set voxel data
            slicer.util.updateVolumeFromArray(labelVolumeNode, maskArray)

            # Import label map into segmentation
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelVolumeNode, segmentationNode
            )

            # Rename the last added segment to the prompt text
            segmentation = segmentationNode.GetSegmentation()
            lastSegmentId = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
            segmentation.GetSegment(lastSegmentId).SetName(prompt)

            # Remove the temporary label map node
            slicer.mrmlScene.RemoveNode(labelVolumeNode)

        # Show segmentation in 3D
        segmentationNode.CreateClosedSurfaceRepresentation()

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

    def test_VoxTellInstantiation(self):
        """Verify that the module can be instantiated and logic created."""
        self.delayDisplay("Testing VoxTell module instantiation")
        logic = VoxTellLogic()
        self.assertIsNotNone(logic)
        self.delayDisplay("VoxTell module instantiation test passed.")
