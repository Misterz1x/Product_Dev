import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.widgets import Cursor 
from scipy.interpolate import interp1d

# --- GLOBAL CONFIGURATION ---
FOLDER_NAME = "txt"
COLUMN_TO_SELECT = 'hip_flexion_r' 
ALL_NORMALIZED_DATA = []
RESAMPLE_POINTS = 101 # Fixed number of points for normalization (0% to 100%)

# ----------------------------------------------------------------------
# --- SNAP CURSOR UTILITY CLASS (Unchanged) ---
# ----------------------------------------------------------------------
class SnappingCursor(Cursor):
    """A cursor that snaps precisely to the nearest data point on the provided line."""
    def __init__(self, ax, line, **kwargs):
        self.line = line
        self.xs = line.get_xdata()
        self.ys = line.get_ydata()
        super().__init__(ax, **kwargs)
        self.target_xy = None 

    def onmove(self, event):
        if event.inaxes != self.ax:
            self.target_xy = None
            return

        x_click = event.xdata
        distances = np.sqrt((self.xs - x_click)**2) 
        closest_index = distances.argmin()
        
        snapped_x = self.xs[closest_index]
        snapped_y = self.ys[closest_index]

        self.target_xy = (snapped_x, snapped_y)
        
        self.vlines.set_xdata([snapped_x])
        
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.vlines)
        self.canvas.blit(self.ax.bbox)

        return False 

    def clear(self, event):
        if self.background:
            self.canvas.restore_region(self.background)
            self.canvas.blit(self.ax.bbox)

# ----------------------------------------------------------------------
# --- GAIT CYCLE SELECTOR CLASS (Logic Unchanged) ---
# ----------------------------------------------------------------------
class GaitCycleSelector:
    
    def __init__(self, data_path, column_to_select):
        self.data_path = data_path
        self.column_to_select = column_to_select
        self.df = None
        self.selected_points = []
        self.fig = None
        self.ax_select = None
        self.file_name = os.path.basename(data_path)
        self.cycle_data = []
        self.cursor = None

        self._load_data()

    def _load_data(self):
        """Loads data from the standard CSV (.txt) file."""
        try:
            print(f"\n--- LOADING DATA: {self.file_name} ---")
            self.df = pd.read_csv(self.data_path, sep=',', decimal='.')
            self.df = self.df.reset_index(drop=True)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"ERROR loading/processing data: {e}")
            self.df = None

    def _on_click(self, event):
        """Handles the click event using the snapping cursor's precise coordinates."""
        if event.inaxes != self.ax_select or not self.cursor or not self.cursor.target_xy:
            return
        
        time_val, data_val = self.cursor.target_xy
        
        match = self.df[
            np.isclose(self.df['time'], time_val) & 
            np.isclose(self.df[self.column_to_select], data_val)
        ]
        
        if match.empty or time_val in [p['time'] for p in self.selected_points]:
            return

        closest_index = match.index[0]
        
        if not self.selected_points:
            label = 'Start'
            color = 'green'
        else:
            label = f'Stop {len(self.selected_points)}'
            color = 'red'
        
        new_point = {'time': time_val, 'index': closest_index, 'label': label}
        self.selected_points.append(new_point)
        
        self.ax_select.plot(time_val, data_val, 'o', color=color, markersize=8)
        self.ax_select.text(time_val, data_val, new_point['label'], fontsize=9, ha='right', color=color)
        self.fig.canvas.draw()
        print(f"Point selected: {new_point['label']} - Time: {time_val:.3f} s (Index: {new_point['index']})")


    def _on_key_press(self, event):
        """Handles key press ('f') to finalize selection."""
        if event.key == 'f':
            print("\n--- FINALIZING POINT SELECTION ---")
            plt.close(self.fig)
            self._process_results()


    def _normalize_cycle(self, start_index, stop_index, cycle_num):
        """Segments, normalizes, and saves an individual cycle."""
        cycle_df = self.df.loc[start_index:stop_index, 
                               ['hip_flexion_r', 'knee_angle_r']].copy()
        
        num_points = len(cycle_df)
        cycle_df['gait_percent'] = np.linspace(0, 100, num_points)
        cycle_df['file'] = self.file_name
        cycle_df['cycle'] = cycle_num
        return cycle_df
    
    
    def _process_results(self):
        """Processes selected points to define and normalize cycles: (P0, P1), (P1, P2), ..."""
        global ALL_NORMALIZED_DATA
        
        if len(self.selected_points) < 2:
            print(f"[{self.file_name}] At least two points (Start and Stop) are required to define a cycle. Skipping.")
            return

        self.selected_points.sort(key=lambda x: x['time'])

        cycle_boundaries = []
        current_start = self.selected_points[0]
        
        for i in range(1, len(self.selected_points)):
            current_stop = self.selected_points[i]
            
            cycle_boundaries.append({
                'Cycle_Number': len(cycle_boundaries) + 1,
                'Start_Index': current_start['index'], 
                'Stop_Index': current_stop['index']
            })
            
            current_start = current_stop 
            
        print(f"\n--- NORMALIZING GAIT CYCLES ({self.file_name}) ---")
        for boundary in cycle_boundaries:
            normalized_cycle = self._normalize_cycle(boundary['Start_Index'], boundary['Stop_Index'], boundary['Cycle_Number'])
            self.cycle_data.append(normalized_cycle)
            print(f"   ✅ Cycle {boundary['Cycle_Number']} normalized (points: {len(normalized_cycle)})")

        if self.cycle_data:
            ALL_NORMALIZED_DATA.extend(self.cycle_data)
        print("--------------------------------------------------\n")
        
        
    def plot_and_select(self):
        """Plots all columns and sets up the interactive selection with Snapping Cursor."""
        if self.df is None: return

        data_cols = [col for col in self.df.columns if col != 'time']
        num_cols = len(data_cols)
        
        self.fig, axes = plt.subplots(num_cols, 1, figsize=(12, 4 * num_cols), sharex=True)
        self.fig.canvas.manager.set_window_title(f'Cycle Selector - {self.file_name}')

        if num_cols == 1: axes = [axes] 
        
        line_to_snap = None
        for i, col in enumerate(data_cols):
            ax = axes[i]
            line, = ax.plot(self.df['time'], self.df[col], label=col)
            ax.set_title(f'{col} vs Time', fontsize=10)
            ax.set_ylabel(col)
            ax.grid(True, linestyle='--')
            
            if col == self.column_to_select:
                self.ax_select = ax
                self.ax_select.set_title(f'SELECT HERE ({self.file_name}): {col} vs Time (SNAP ON)', color='blue', fontweight='bold', fontsize=12)
                line_to_snap = line

        axes[-1].set_xlabel('Time (s)')
        
        if self.ax_select and line_to_snap:
            self.cursor = SnappingCursor(self.ax_select, line_to_snap, useblit=True, color='red', linewidth=1, horizOn=False)
            
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            plt.tight_layout()
            
            print("\n=======================================================")
            print(f" Processing file: {self.file_name} ")
            print("-------------------------------------------------------")
            print("1. **First click (GREEN)**: Sets the Start of Cycle 1.")
            print("2. **Subsequent clicks (RED)**: Set the Stop of the previous cycle (and Start of the next).")
            print("3. Press **'f'** key to finalize selection.")
            print("=======================================================\n")
            
            plt.show() 

# ----------------------------------------------------------------------
# --- REMUESTREO AUXILIAR ---
# ----------------------------------------------------------------------

def resample_cycle(cycle_df, column, resample_points):
    """
    Interpola una columna del ciclo a un número fijo de puntos (0% a 100%).
    """
    x_old = cycle_df['gait_percent'].values
    y_old = cycle_df[column].values
    
    # Crear la función de interpolación
    interp_func = interp1d(x_old, y_old, kind='linear')
    
    # Nuevo eje X (0 a 100 con RESAMPLE_POINTS)
    x_new = np.linspace(0, 100, resample_points)
    
    # Nuevo eje Y (valores interpolados)
    y_new = interp_func(x_new)
    
    return y_new

# ----------------------------------------------------------------------
# --- FINAL PLOT FUNCTION (RIGHT KNEE STATISTICS - ALIGNED) ---
# ----------------------------------------------------------------------

def plot_normalized_cycles_by_file(all_data):
    """
    Plots the normalized Right Knee Angle for all files, showing individual cycles, mean, and STD,
    ensuring all cycles are perfectly aligned via resampling before calculating stats.
    """
    if not all_data:
        print("No normalized cycles to plot.")
        return

    df_all_cycles = pd.concat(all_data, ignore_index=True)
    unique_files = df_all_cycles['file'].unique()
    
    # Puntos fijos para el remuestreo
    resample_x = np.linspace(0, 100, RESAMPLE_POINTS)
    
    for file_name in unique_files:
        df_file = df_all_cycles[df_all_cycles['file'] == file_name]
        cycle_groups = df_file.groupby('cycle')
        
        # Almacena los valores de Knee Angle (derecha) remuestreados para el cálculo de estadísticas
        resampled_cycles = []

        # 1. Remuestrear (Resample) todos los ciclos
        for cycle_num, group in cycle_groups:
            resampled_y = resample_cycle(group, 'knee_angle_r', RESAMPLE_POINTS)
            resampled_cycles.append(resampled_y)

        # Convertir a array de NumPy para cálculos matriciales
        resampled_array = np.array(resampled_cycles)

        # 2. Calcular Media y Desviación Estándar (STD)
        y_mean = np.mean(resampled_array, axis=0)
        y_std = np.std(resampled_array, axis=0)
        
        # --- Configuración del Plot ---
        fig, ax_knee = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        fig.canvas.manager.set_window_title(f'Normalized Cycles (Right Knee): {file_name}')
        fig.suptitle(f'Normalized Right Knee Angle ({file_name})', fontsize=14)

        # 3. Plotear Desviación Estándar (STD) Área (Light Blue Light)
        ax_knee.fill_between(
            resample_x, 
            y_mean - y_std, 
            y_mean + y_std, 
            color='lightskyblue', 
            alpha=0.4, 
            label='Std Dev $(\pm 1\sigma)$'
        )

        # 4. Plotear Media (Dark Blue)
        ax_knee.plot(
            resample_x, 
            y_mean, 
            color='darkblue', 
            linewidth=3, 
            label='Mean Angle'
        )

        # 5. Plotear Ciclos Individuales (Light Blue)
        # Usamos el array remuestreado para el ploteo individual (más limpio)
        is_first_cycle = True
        for y_data in resampled_array:
            label = 'Individual Cycles' if is_first_cycle else "_nolegend_"
            ax_knee.plot(
                resample_x, 
                y_data, 
                color='lightblue', 
                linestyle='--', 
                linewidth=1, 
                alpha=0.6,
                label=label
            )
            is_first_cycle = False 

        ax_knee.set_title('Right Knee Angle with Statistics (Aligned)', fontsize=12)
        ax_knee.set_ylabel('Angle (degrees)')
        ax_knee.set_xlabel('Gait Cycle Percentage (%)')
        ax_knee.grid(True, linestyle='--')
        ax_knee.legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# ----------------------------------------------------------------------
# --- MAIN FUNCTION (Unchanged) ---
# ----------------------------------------------------------------------

def process_all_files(folder):
    """Iterates over all .txt files in the specified folder."""
    
    global ALL_NORMALIZED_DATA
    ALL_NORMALIZED_DATA = [] 

    if not os.path.isdir(folder):
        print(f"ERROR: Folder '{folder}' does not exist. Run the .mot conversion script first.")
        return

    all_files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
    
    if not all_files:
        print(f"No .txt files found in folder '{folder}'.")
        return

    print(f"Found {len(all_files)} .txt files to process.")

    for file_name in all_files:
        full_path = os.path.join(folder, file_name)
        
        selector = GaitCycleSelector(full_path, COLUMN_TO_SELECT)
        if selector.df is not None:
            selector.plot_and_select()
        
    print("\n=======================================================")
    print("✅ Selection process completed for all files.")
    print("=======================================================")
    
    plot_normalized_cycles_by_file(ALL_NORMALIZED_DATA)
    print("\n🎉 Final normalized plot (Right Knee Only) completed.")

if __name__ == "__main__":
    process_all_files(FOLDER_NAME)