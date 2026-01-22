#from src import residual_plot
import pandas as pd
from sklearn.model_selection import train_test_split

#from obj_func.maglearn_utility import scale_temp, sphere_transform_df
#from obj_func.kendall import BnnHa, BnnMs, BnnPrice

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np



def _validate_dir(directory):
    assert isinstance(directory, str) or isinstance(directory, Path), f"directory be a string or a pathlib.Path!"
    print(f"Create Path object from directory...")
    directory = Path(directory).absolute()
    assert directory.exists(), f"directory does not exist: '{directory}'"
    return directory


# ====================================================================
# Plotting functions extracted from Jupyter notebook
# ====================================================================



from pathlib import Path
from typing import Iterable, List, Optional
import matplotlib.pyplot as plt

def save_plot(
    filename: str,
    dir_name: Optional[Path] = None,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 300,
    tight: bool = True,
    transparent: bool = False
) -> List[Path]:
    """
    Save the current Matplotlib figure to one or more formats (e.g., PNG + SVG).

    Parameters
    ----------
    filename : str
        Base filename (with or without extension). If an extension is present,
        it will be replaced by each requested format.
    dir_name : Path or None
        Directory to save into. If None, saves into the current working directory.
        If the directory does not exist, it will be created.
    formats : Iterable[str]
        Iterable of formats to save, e.g. ("png", "svg"). Case-insensitive.
    dpi : int
        DPI for raster outputs (e.g., PNG). Ignored for vector formats like SVG/PDF.
    tight : bool
        If True, uses bbox_inches='tight' to avoid clipping.
    transparent : bool
        If True, save with transparent background.
    
    Returns
    -------
    List[Path]
        List of file paths that were saved.
    """
    # Normalize dir_name -> Path and ensure directory exists
    if dir_name is None:
        dir_name = Path.cwd()
    elif not isinstance(dir_name, Path):
        dir_name = Path(dir_name)
    dir_name.mkdir(parents=True, exist_ok=True)

    # Normalize formats and validate
    formats = [str(fmt).lower().strip() for fmt in formats]
    valid_formats = {"png", "svg", "pdf", "jpg", "jpeg", "tif", "tiff"}
    for fmt in formats:
        if fmt not in valid_formats:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {sorted(valid_formats)}")

    # Base name without any extension, so we rebuild per format
    base = Path(filename).stem
    saved_paths: List[Path] = []

    # Common save kwargs
    save_kwargs = {"transparent": transparent}
    if tight:
        save_kwargs["bbox_inches"] = "tight"

    for fmt in formats:
        out_path = dir_name / f"{base}.{fmt}"
        # Raster formats -> use DPI
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            plt.savefig(out_path, dpi=dpi, **save_kwargs)
        else:
            # Vector formats ignore DPI
            plt.savefig(out_path, **save_kwargs)
        print(f"Saved plot to {out_path}")
        saved_paths.append(out_path)





def plot_cv_predictions(y_pred, y_true, std, title, filename, dirname, figsize: tuple=None, std_color="C0"):
    """
    Plots residuals and measured values vs predicted values in two stacked subplots.
    The std parameter is optional and will be plotted as errorbars if provided. 
    """

    # Figure setup
    FIGSIZE = (10, 8)
    if figsize is None:
        figsize = FIGSIZE
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title + f"\n$R^2$ = {r2_score(y_true, y_pred):.2f} | N = {len(y_true)}", fontsize="medium")

    # -------------------- Upper plot --------------------

    residuals = y_true - y_pred
    axs[0].scatter(y_pred, residuals, alpha=0.4, edgecolor='k', facecolor='none', s=20, zorder=20, marker="h")
    axs[0].axhline(0.0, color='red', linestyle='--', lw=1.3, alpha=0.6, label='0.0', zorder=10)
    # looks 
    axs[0].grid(alpha=0.5, zorder=0)
    axs[0].legend(loc='lower left', fontsize=10)
    axs[0].set_ylabel("Residual (T)", fontsize=20)

    # -------------------- Lower plot --------------------

    max_y = max(y_true.max(), y_pred.max()) * 1.1
    axs[1].plot([-max_y, max_y], [-max_y, max_y], 'r--', lw=1.3, alpha=0.6, label="ideal fit", zorder=0)

    axs[1].scatter(y_pred, y_true, alpha=0.4, edgecolor='k', facecolor='none', s=20, marker="h", label="mean", zorder=20)
    if std is not None: 
        axs[1].errorbar(y_pred, y_true, xerr=std, fmt='s', markersize=0, elinewidth=1.6, color=std_color, alpha=0.3, label="$\sigma$", zorder=10)

    axs[1].grid(alpha=0.5, zorder=0)
    axs[1].legend(loc='upper left', fontsize=10)
    axs[1].set_ylabel("Measured (T)", fontsize=20)
    axs[1].set_xlabel("Predicted (T)", fontsize=20)

    # limit axes to min and max of y_true and y_pred
    axs[1].set_xlim(y_pred.min()*0.9, y_pred.max()*1.1)
    axs[1].set_ylim(y_true.min()*0.9, y_true.max()*1.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    save_plot(filename, dir_name=dirname)
    plt.show()



def plot_cv_predictions_color_std(y_pred, y_true, std, title, details, filename, dirname):
    """
    Plots residuals and measured values vs predicted values using 5-fold CV.
    The upper plot uses colors to represent std values and includes a colorbar.

    Parameters:
    - y_pred: predicted values (array-like)
    - y_true: true values (array-like)
    - std: standard deviation values for predictions (array-like)
    - title: plot title
    - filename: file name to save the plot
    - dirname: directory name to save the plot
    - figsize: tuple, optional, figure size
    - cmap: colormap for std values in upper plot
    """
    # Figure setup
    #FIGfigsizeSIZE = (10, 8)
    figsize = (8, 7)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    TICKLABELSIZE = 14  
    LABELSIZE = 20 
    TITLESIZE = 20 

    # TITLES
    # small detail line
    details = details + f" - $R^2$ = {r2_score(y_true, y_pred):.2f} | N = {len(y_true)}\n"
    fig.text(
        0.5, 1, 
        details,
        #transform=axs[0].transAxes,
        fontsize=TITLESIZE/2,
        ha="center", va="bottom"
    )

    # big bold main title below it
    fig.text(
        0.5, .97, 
        title,
        fontsize=TITLESIZE, 
        ha="center", va="bottom"
    )


    # -------------------- Upper plot --------------------

    residuals = y_true - y_pred

    sc = axs[0].scatter(
        y_pred, residuals,
        c=std,
        cmap="viridis_r",
        alpha=0.6,
        edgecolor="#555555",  # dark grey
        linewidth=0.4,   # controls thickness of edge
        s=70,  
        marker="h",
        zorder=20
    )

    # Create a dedicated colorbar axes ABOVE the upper plot
    bbox = axs[0].get_position()
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([bbox.x0    -0.02,       # left
                            bbox.y1    +0.03,       # bottom
                            bbox.width +0.09,       # width
                                        0.05])      # height

    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    TOP = False
    BOTTOM = True
    cbar.ax.tick_params(labeltop=TOP, labelbottom=BOTTOM, top=TOP, bottom=BOTTOM, 
                        labelsize=TICKLABELSIZE,
                        direction="inout", size=5) # hide ticks with size=0
    cbar.set_label(loc="center", label="Uncertainty $\sigma$", 
                   labelpad=-40, fontsize=LABELSIZE*0.75)

    axs[0].axhline(0.0, color='red', linestyle='--', lw=1.3, alpha=0.6, zorder=10)
    axs[0].grid(alpha=0.5, zorder=0)
    axs[0].set_ylabel("Residual (T)", fontsize=20)

    # -------------------- Lower plot --------------------

    max_y = max(y_true.max(), y_pred.max()) * 1.1
    axs[1].plot([-max_y, max_y], [-max_y, max_y], 'r--', lw=1.3, alpha=0.6, label="ideal fit", zorder=0)

    axs[1].scatter(
        y_pred, 
        y_true, 
        alpha=0.4, 
        edgecolor="black",
        linewidth=1.3,   # controls thickness of edge
        facecolor='none', 
        s=50, 
        marker="h", 
        zorder=20
    )
    axs[1].grid(alpha=0.5, zorder=0)
    axs[1].legend(loc='upper left', fontsize=15)
    axs[1].set_ylabel("Measured (T)", fontsize=LABELSIZE)
    axs[1].set_xlabel("Predicted (T)", fontsize=LABELSIZE)

    # -------------------- Final adjustments -------------------

    # limit axes to min and max of y_true and y_pred
    axs[1].set_xlim(y_pred.min()*0.7, y_pred.max()*1.1)
    axs[1].set_ylim(y_true.min()*0.7, y_true.max()*1.1)

    fig.align_ylabels(axs)
    for ax in axs: 
        ax.tick_params(labelsize=TICKLABELSIZE)

    # Adjust for suptitle
    plt.tight_layout(rect=[0,       # left
                           0,       # bottom
                           1,       # right
                           0.88])    # top
    save_plot(filename, dir_name=dirname)
    plt.show()



# ====================================================================


def plot_loss(history: dict, filename, dirname, figsize: tuple=None): 
    FIGSIZE = (8, 5)
    if figsize is None:
        figsize = FIGSIZE
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss (Gaussian NLL)', fontsize=20)
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(alpha=0.5, zorder=0)
    save_plot(filename, dir_name=dirname)
    plt.show()


def plot_val_score_vs_epochs(history: dict, filename, dirname, figsize: tuple=None): 
    """ will this work? """
    FIGSIZE = (8, 5)
    if figsize is None:
        figsize = FIGSIZE
    plt.figure(figsize=figsize)
    plt.plot(history.history['score'], label='Training Score')
    if 'val_score' in history.history:
        plt.plot(history.history['val_score'], label='Validation Score')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.title('Training and Validation Score over Epochs')
    plt.legend()
    plt.grid(alpha=0.5, zorder=0)
    save_plot(filename, dir_name=dirname)
    plt.show()
