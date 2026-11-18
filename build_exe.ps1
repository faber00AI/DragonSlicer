# Portable build: dist\DragonSlicer.exe: GUI, plus `DragonSlicer.exe cli ...` and `DragonSlicer.exe mcp`. Needs: pip install pyinstaller
$R = $PSScriptRoot
$common = @("--noconfirm", "--clean", "--onefile", "--distpath", "$R\dist", "--paths", $R,
    "--add-data", "$R\machines;machines", "--add-data", "$R\presets;presets", "--add-data", "$R\library;library", "--add-data", "$R\assets;assets",
    "--collect-all", "trimesh", "--collect-submodules", "skimage", "--collect-binaries", "manifold3d",
    "--collect-all", "pymeshfix", "--collect-all", "fast_simplification",
    "--hidden-import", "scipy.spatial.transform._rotation_groups", "--hidden-import", "mapbox_earcut",
    "--exclude-module", "matplotlib", "--exclude-module", "tkinter", "--exclude-module", "IPython", "--exclude-module", "pytest")
python -m PyInstaller @common --windowed --icon "$R\assets\dragonslide.ico" --name DragonSlicer --workpath "$R\build\gui" --specpath "$R\build" `
    --hidden-import dragonslide_cli --hidden-import dragonslide_mcp --collect-all mcp "$R\main.py"
Remove-Item -Recurse -Force "$R\build"
