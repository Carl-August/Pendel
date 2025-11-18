#!/usr/bin/env python3
"""
Video Dark Pixel Analyzer
Erkennt dunkle Pixel in Videos im Vergleich zum Hintergrund
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from pathlib import Path
import sys
import math
import pickle

def setup_matplotlib():
    """Setup matplotlib for proper rendering"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

class VideoDarkPixelAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialisierung der Variablen
        self.background_region = None
        self.dark_region = None
        self.background_threshold = None
        self.results = []
        self.analysis_region = None  # Neuer: Bereich für Analyse dunkler Pixel
        
    def select_analysis_region(self, frame, window_name="Analyse-Bereich Auswahl"):
        """
        Ermöglicht die Auswahl eines Bereichs mit Zoom-Funktion (Mausrad).
        Der Benutzer kann zoomen und dann einen rechteckigen Bereich auswählen.
        """
        print(f"\nBitte wählen Sie den Analyse-Bereich aus:")
        print("- Mausrad: Zoom rein/raus")
        print("- Linke Maustaste: Rechteckigen Bereich auswählen")
        print("- Enter/SPACE: Auswahl bestätigen")
        
        display_frame = frame.copy()
        
        # Verwende SelectROI für rechteckige Auswahl (builtin zoom möglich)
        roi = cv2.selectROI(window_name, display_frame, showCrosshair=True)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:  # Prüfe ob eine gültige Region ausgewählt wurde
            return roi
        else:
            raise ValueError("Kein gültiger Analyse-Bereich ausgewählt")
    
    def select_region(self, frame, region_name, window_name="Region Selection"):
        """
        Ermöglicht die Auswahl einer Region im Frame
        """
        print(f"\nBitte wählen Sie die {region_name} aus:")
        print("1. Klicken Sie links oben auf die Region")
        print("2. Ziehen Sie eine rechteckige Auswahl")
        print("3. Drücken Sie Enter oder SPACE um zu bestätigen")
        
        # Erstelle kopie für die Auswahl
        display_frame = frame.copy()
        
        # Verwende SelectROI für rechteckige Auswahl
        roi = cv2.selectROI(window_name, display_frame, showCrosshair=True)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:  # Prüfe ob eine gültige Region ausgewählt wurde
            return roi
        else:
            raise ValueError(f"Keine gültige {region_name} ausgewählt")
    
    def select_pixels(self, frame, pixel_name, window_name="Pixel Selection"):
        """
        Ermöglicht die Auswahl einzelner Pixel durch Mausklick.
        Der Benutzer kann mehrere Pixel klicken. Beendet mit ESC oder SPACE.
        Gibt eine Liste von (x, y) Koordinaten zurück.
        """
        print(f"\nBitte wählen Sie Pixel für '{pixel_name}' aus:")
        print("- Linke Maustaste: Pixel hinzufügen")
        print("- ESC oder SPACE: Fertig")
        
        display_frame = frame.copy()
        pixels = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixels.append((x, y))
                # Zeichne einen Kreis um den ausgewählten Pixel
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(window_name, display_frame)
                print(f"  ✓ Pixel hinzugefügt: ({x}, {y})")
        
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, display_frame)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Warte auf Benutzereingabe (ESC=27, SPACE=32)
        while True:
            key = cv2.waitKey(0)
            if key == 27 or key == 32:  # ESC or SPACE
                break
        
        cv2.destroyAllWindows()
        
        if not pixels:
            raise ValueError(f"Keine Pixel für '{pixel_name}' ausgewählt")
        
        return pixels
    
    def extract_pixels_intensity(self, frame, pixels, analysis_roi=None):
        """
        Extrahiert die mittlere Intensität von ausgewählten Pixeln.
        pixels: Liste von (x, y) Koordinaten
        Falls analysis_roi gesetzt ist, werden nur Pixel innerhalb berücksichtigt.
        """
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        valid_pixels = []
        
        for x, y in pixels:
            # Prüfe ob Pixel im Frame liegt
            if 0 <= y < gray_frame.shape[0] and 0 <= x < gray_frame.shape[1]:
                # Prüfe ob Pixel in analysis_roi liegt (falls gesetzt)
                if analysis_roi is not None:
                    ax, ay, aw, ah = analysis_roi
                    if not (ax <= x < ax + aw and ay <= y < ay + ah):
                        continue
                
                valid_pixels.append(gray_frame[y, x])
        
        if not valid_pixels:
            return np.nan
        
        return np.mean(valid_pixels)
    
    def extract_region_intensity(self, frame, roi, analysis_roi=None):
        """
        Extrahiert die mittlere Intensität einer Region.
        Falls analysis_roi gesetzt ist, wird nur der Überlappungsbereich berücksichtigt.
        """
        x, y, w, h = roi
        
        # Falls ein Analyse-Bereich definiert ist, beschneide die Region auf diesen Bereich
        if analysis_roi is not None:
            ax, ay, aw, ah = analysis_roi
            # Berechne Überlappung
            x1 = max(x, ax)
            y1 = max(y, ay)
            x2 = min(x + w, ax + aw)
            y2 = min(y + h, ay + ah)
            
            if x2 <= x1 or y2 <= y1:
                # Keine Überlappung
                return np.nan
            
            region = frame[y1:y2, x1:x2]
        else:
            region = frame[y:y+h, x:x+w]
        
        # Konvertiere zu Graustufen falls nötig
        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region
            
        return np.mean(gray_region)
    
    def analyze_video(self, background_roi, dark_roi, intensity_threshold=50, analysis_roi=None, 
                      background_pixels=None, dark_pixels=None):
        """
        Analysiert das Video und erkennt dunkle Pixel.
        Kann entweder mit ROI (Region of Interest) oder mit einzelnen Pixeln arbeiten.
        
        Falls background_pixels/dark_pixels gesetzt sind, werden diese statt ROI verwendet.
        Falls analysis_roi gesetzt ist, werden nur Pixel innerhalb dieses Bereichs analysiert.
        """
        print(f"\nAnalysiere Video mit {self.total_frames} Frames ({self.duration:.2f}s)...")
        print(f"FPS: {self.fps}")
        if analysis_roi:
            print(f"Analyse-Bereich: {analysis_roi}")
        
        # Bestimme Analyse-Modus
        use_pixels = background_pixels is not None and dark_pixels is not None
        
        if use_pixels:
            print(f"Modus: Pixel-basierte Analyse")
            print(f"- Hintergrund-Pixel: {len(background_pixels)} Pixel")
            print(f"- Dunkle Pixel: {len(dark_pixels)} Pixel")
        else:
            print(f"Modus: ROI-basierte Analyse")
            print(f"- Hintergrund-ROI: {background_roi}")
            print(f"- Dunkle ROI: {dark_roi}")
        
        frame_count = 0
        last_dark_frame = None
        dark_frames = []
        recent_time_diffs = []  # Speichere die letzten 5 Zeitdifferenzen
        
        # Determine progress update interval (update ~100 times)
        try:
            total = max(1, int(self.total_frames))
        except Exception:
            total = 1
        update_interval = max(1, total // 100)
        
        # Berechne Hintergrund-Referenzintensität vom ersten Frame
        first_frame = self.get_frame(0)
        if first_frame is None:
            raise ValueError("Kann ersten Frame nicht lesen")
        
        if use_pixels:
            background_reference = self.extract_pixels_intensity(first_frame, background_pixels, analysis_roi)
        else:
            background_reference = self.extract_region_intensity(first_frame, background_roi, analysis_roi)
        
        print(f"Hintergrund-Referenzintensität: {background_reference:.2f}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Analysiere aktuellen Frame
            if use_pixels:
                background_intensity = self.extract_pixels_intensity(frame, background_pixels, analysis_roi)
                dark_intensity = self.extract_pixels_intensity(frame, dark_pixels, analysis_roi)
            else:
                background_intensity = self.extract_region_intensity(frame, background_roi, analysis_roi)
                dark_intensity = self.extract_region_intensity(frame, dark_roi, analysis_roi)
            
            # Prüfe auf dunkle Pixel
            intensity_difference = background_intensity - dark_intensity
            
            # frame_time is calculated from the provided time_fps if set,
            # otherwise fall back to the video's fps
            time_fps = getattr(self, 'used_recorded_fps', None) or self.fps
            frame_time = frame_count / time_fps
            is_dark_frame = intensity_difference > intensity_threshold
            
            if is_dark_frame:
                dark_frames.append({
                    'frame_number': frame_count,
                    'time_seconds': frame_time,
                    'background_intensity': background_intensity,
                    'dark_intensity': dark_intensity,
                    'intensity_difference': intensity_difference
                })
                
                # Berechne Zeitdifferenz zum vorherigen dunklen Frame
                # Nur speichern, wenn größer als 1/fps (d.h. nicht direkt nacheinander)
                if last_dark_frame is not None:
                    time_diff = frame_time - last_dark_frame
                    min_time_gap = 1.0 / time_fps  # Minimale Zeitlücke: 1 Frame
                    
                    if time_diff >= min_time_gap:
                        # Neue "Gruppe" von dunklen Frames erkannt
                        dark_frames[-1]['time_difference_to_previous'] = time_diff
                        recent_time_diffs.append(time_diff)
                        if len(recent_time_diffs) > 5:
                            recent_time_diffs.pop(0)
                        # Drucke Meldung und dann Progressbar darunter
                        #print(f"\nDunkles Frame bei {frame_time:.3f}s (Frame {frame_count}) - "
                        #      f"Zeitdifferenz: {time_diff:.3f}s | Letzte: " + 
                        #     ", ".join([f"{d:.3f}s" for d in recent_time_diffs[-5:]]))
                        last_dark_frame = frame_time
                    else:
                        # Rahmen sind direkt hintereinander (Teil der gleichen Gruppe)
                        pass
                else:
                    #print(f"\nErstes dunkles Frame bei {frame_time:.3f}s (Frame {frame_count})")
                    last_dark_frame = frame_time
            
            frame_count += 1
            
            # Fortschrittsanzeige mit letzten 5 Zeitdifferenzen
            if frame_count % update_interval == 0 or frame_count == total:
                self._print_progress(frame_count, total, recent_time_diffs)
        
        self.results = dark_frames
        return dark_frames

    def _print_progress(self, current, total, recent_diffs=None, bar_length=40):
        """Terminal-Progressbar mit letzten 5 Zeitdifferenzen"""
        try:
            percent = float(current) / float(total)
        except Exception:
            percent = 0.0
        filled = int(round(bar_length * percent))
        bar = '#' * filled + '-' * (bar_length - filled)
        
        # Formatiere letzte Zeitdifferenzen
        diffs_str = ""
        if recent_diffs:
            diffs_str = " | Letzte Zeiten: " + ", ".join([f"{d:.3f}s" for d in recent_diffs[-5:]])
        
        print(f"\rProgress: |{bar}| {percent*100:6.2f}% ({current}/{total}){diffs_str}", end='', flush=True)
        if current >= total:
            print()

    def get_frame(self, frame_number):
        """
        Liest einen spezifischen Frame
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset Position
        return frame if ret else None
    
    def save_results(self, output_path):
        """
        Speichert die Ergebnisse in einer JSON-Datei
        """
        results_data = {
            'video_info': {
                'video_path': str(self.video_path),
                'fps': self.fps,
                'used_recorded_fps': getattr(self, 'used_recorded_fps', self.fps),
                'total_frames': self.total_frames,
                'duration_seconds': self.duration,
                'frame_dimensions': (self.frame_width, self.frame_height)
            },
            'analysis_results': self.results,
            'summary': {
                'total_dark_frames': len(self.results),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nErgebnisse gespeichert in: {output_path}")

    def save_time_differences(self, out_dir=None, remove_outliers=True):
        """
        Speichert nur die 'time_difference_to_previous' Werte in einer Textdatei
        im Ordner 'Ergebnisse' (je Videodatei eine Datei).
        Falls remove_outliers=True, werden Ausreißer entfernt (Werte > 2 Standardabweichungen vom Mittelwert).
        """
        if not self.results:
            print("Keine Ergebnisse vorhanden, nichts zu speichern.")
            return

        try:
            # Bestimme Ausgabeverzeichnis (default: 'Ergebnisse' neben Skript)
            if out_dir is None:
                out_dir = Path(__file__).parent / 'Ergebnisse'
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            base = Path(self.video_path).stem if self.video_path else 'video'
            out_file = out_dir / f"{base}_time_differences.txt"

            # Sammle time_difference_to_previous Werte (>= 1 Sekunde)
            time_diffs = []
            for r in self.results:
                td = r.get('time_difference_to_previous')
                if td is not None and td >= 1.0:
                    time_diffs.append(td)
            
            if not time_diffs:
                print("Keine Zeitdifferenzen vorhanden.")
                return
            
            # Ausreißer entfernen (optional)
            if remove_outliers and len(time_diffs) > 2:
                mean_val = np.mean(time_diffs)
                std_val = np.std(time_diffs)
                threshold = 2.0  # 2 Standardabweichungen
                filtered_diffs = [td for td in time_diffs if abs(td - mean_val) <= threshold * std_val]
                
                removed_count = len(time_diffs) - len(filtered_diffs)
                if removed_count > 0:
                    print(f"Ausreißer entfernt: {removed_count} Wert(e)")
                    time_diffs = filtered_diffs
            
            # Formatiere und speichere Werte (gerundet auf 5 Dezimalstellen, Komma als Trennzeichen)
            lines = []
            for td in time_diffs:
                rounded_td = round(td, 5)
                formatted_td = str(rounded_td).replace('.', ',')
                lines.append(f"{formatted_td}\n")

            with open(out_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"Zeitdifferenzen gespeichert in: {out_file} ({len(lines)} Einträge)")
        except Exception as e:
            print(f"Fehler beim Speichern der Zeitdifferenzen: {e}")

    def save_every_second_result(self, out_dir=None, remove_outliers=True):
        """
        Speichert nur jeden zweiten 'time_difference_to_previous' Wert (Index 1, 3, 5, ...) in einer Textdatei.
        Falls remove_outliers=True, werden Ausreißer entfernt (Werte > 2 Standardabweichungen vom Mittelwert).
        """
        if not self.results:
            print("Keine Ergebnisse vorhanden, nichts zu speichern.")
            return

        try:
            # Bestimme Ausgabeverzeichnis (default: 'Ergebnisse' neben Skript)
            if out_dir is None:
                out_dir = Path(__file__).parent / 'Ergebnisse'
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            base = Path(self.video_path).stem if self.video_path else 'video'
            out_file = out_dir / f"{base}_every_second_time_differences.txt"

            # Sammle time_difference_to_previous Werte (>= 1 Sekunde)
            time_diffs = []
            for r in self.results:
                td = r.get('time_difference_to_previous')
                if td is not None and td >= 1.0:
                    time_diffs.append(td)
            
            if not time_diffs:
                print("Keine Zeitdifferenzen vorhanden.")
                return
            
            # Wähle nur jeden zweiten Wert (Index 1, 3, 5, ...)
            every_second_diffs = time_diffs[1::2]
            
            if not every_second_diffs:
                print("Keine zweiten Ergebnisse vorhanden (zu wenig Einträge).")
                return
            
            # Ausreißer entfernen (optional)
            if remove_outliers and len(every_second_diffs) > 2:
                mean_val = np.mean(every_second_diffs)
                std_val = np.std(every_second_diffs)
                threshold = 2.0  # 2 Standardabweichungen
                filtered_diffs = [td for td in every_second_diffs if abs(td - mean_val) <= threshold * std_val]
                
                removed_count = len(every_second_diffs) - len(filtered_diffs)
                if removed_count > 0:
                    print(f"Ausreißer entfernt (zweite Ergebnisse): {removed_count} Wert(e)")
                    every_second_diffs = filtered_diffs
            
            # Formatiere und speichere Werte (gerundet auf 5 Dezimalstellen, Komma als Trennzeichen)
            lines = []
            for td in every_second_diffs:
                rounded_td = round(td, 5)
                formatted_td = str(rounded_td).replace('.', ',')
                lines.append(f"{formatted_td}\n")

            with open(out_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"Zweite Ergebnisse gespeichert in: {out_file} ({len(lines)} Einträge)")
        except Exception as e:
            print(f"Fehler beim Speichern der zweiten Ergebnisse: {e}")

    def save_double_period_differences(self, out_dir=None, remove_outliers=True):
        """
        Speichert die kumulierten Zeitdifferenzen zwischen alternierenden 'time_difference_to_previous' Werten.
        Das heißt: Zeit(0) + Zeit(1), Zeit(2) + Zeit(3), etc.
        Dies entspricht der doppelten Periode eines Pendels (volle Hin- und Rückbewegung).
        Falls remove_outliers=True, werden Ausreißer entfernt (Werte > 2 Standardabweichungen vom Mittelwert).
        """
        if not self.results:
            print("Keine Ergebnisse vorhanden, nichts zu speichern.")
            return

        try:
            # Bestimme Ausgabeverzeichnis (default: 'Ergebnisse' neben Skript)
            if out_dir is None:
                out_dir = Path(__file__).parent / 'Ergebnisse'
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            base = Path(self.video_path).stem if self.video_path else 'video'
            out_file = out_dir / f"{base}_double_period_differences.txt"

            # Sammle time_difference_to_previous Werte (>= 1 Sekunde)
            time_diffs = []
            for r in self.results:
                td = r.get('time_difference_to_previous')
                if td is not None and td >= 1.0:
                    time_diffs.append(td)
            
            if not time_diffs or len(time_diffs) < 2:
                print("Zu wenige Zeitdifferenzen vorhanden für doppelte Periode.")
                return
            
            # Berechne doppelte Perioden: (Zeit[0] + Zeit[1]), (Zeit[2] + Zeit[3]), ...
            double_periods = []
            for i in range(0, len(time_diffs) - 1, 2):
                double_period = time_diffs[i] + time_diffs[i + 1]
                double_periods.append(double_period)
            
            if not double_periods:
                print("Keine doppelten Perioden berechnet.")
                return
            
            # Ausreißer entfernen (optional)
            if remove_outliers and len(double_periods) > 2:
                mean_val = np.mean(double_periods)
                std_val = np.std(double_periods)
                threshold = 2.0  # 2 Standardabweichungen
                filtered_periods = [dp for dp in double_periods if abs(dp - mean_val) <= threshold * std_val]
                
                removed_count = len(double_periods) - len(filtered_periods)
                if removed_count > 0:
                    print(f"Ausreißer entfernt (doppelte Perioden): {removed_count} Wert(e)")
                    double_periods = filtered_periods
            
            # Formatiere und speichere Werte (gerundet auf 5 Dezimalstellen, Komma als Trennzeichen)
            lines = []
            for dp in double_periods:
                rounded_dp = round(dp, 5)
                formatted_dp = str(rounded_dp).replace('.', ',')
                lines.append(f"{formatted_dp}\n")

            with open(out_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"Doppelte Perioden gespeichert in: {out_file} ({len(lines)} Einträge)")
        except Exception as e:
            print(f"Fehler beim Speichern der doppelten Perioden: {e}")

    def calculate_gravity(self, pendulum_length, half_period):
        """
        Berechnet die Gravitationskonstante g aus der Pendellänge und Halbperiode.
        Formel: g = 4*π²*L / T²
        wobei L die Pendellänge ist und T die Periodendauer (2 * half_period)
        """
        T = 2 * half_period  # Periodendauer = 2 * Halbperiode
        g = (4 * math.pi**2 * pendulum_length) / (T**2)
        return g

    def print_summary(self):
        """
        Gibt eine Zusammenfassung der Ergebnisse aus
        """
        if not self.results:
            print("Keine dunklen Frames gefunden.")
            return
            
        print(f"\n=== ANALYSE ZUSAMMENFASSUNG ===")
        print(f"Gefundene dunkle Frames: {len(self.results)}")
        print(f"Video Dauer: {self.duration:.2f} Sekunden")
        
        # Sammle nur Frames mit time_difference_to_previous
        time_diffs = [r['time_difference_to_previous'] for r in self.results if 'time_difference_to_previous' in r]
        
        if len(time_diffs) > 0:
            avg_interval = np.mean(time_diffs)
            print(f"Durchschnittliche Zeitdifferenz zwischen dunklen Frames: {avg_interval:.3f}s")
            
            if len(time_diffs) > 1:
                std_interval = np.std(time_diffs)
                print(f"Standardabweichung der Zeitdifferenzen: {std_interval:.3f}s")
        
        print(f"\nDetaillierte Zeitdifferenzen:")
        for i, result in enumerate(self.results):
            if 'time_difference_to_previous' in result:
                time_diff = result['time_difference_to_previous']
                print(f"Frame {i+1}: {result['time_seconds']:.3f}s (+{time_diff:.3f}s)")
            else:
                print(f"Frame {i+1}: {result['time_seconds']:.3f}s (Start)")
    
    def close(self):
        """
        Schließt die Video-Capture-Verbindung
        """
        self.cap.release()
        cv2.destroyAllWindows()

def find_videos_in_dir(video_dir):
    """
    Liefert eine Liste von Video-Dateien im angegebenen Ordner zurück.
    Unterstützte Endungen: .mp4, .avi, .mov, .mkv
    """
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    p = Path(video_dir)
    if not p.exists() or not p.is_dir():
        return []
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    return files

def save_last_settings(settings, settings_file=None):
    """
    Speichert die letzten Einstellungen in einer pickle-Datei.
    """
    if settings_file is None:
        settings_file = Path(__file__).parent / '.last_settings.pkl'
    
    try:
        with open(settings_file, 'wb') as f:
            pickle.dump(settings, f)
    except Exception as e:
        print(f"Warnung: Konnte Einstellungen nicht speichern: {e}")

def load_last_settings(settings_file=None):
    """
    Lädt die letzten Einstellungen aus einer pickle-Datei.
    Gibt None zurück, falls die Datei nicht existiert oder nicht lesbar ist.
    """
    if settings_file is None:
        settings_file = Path(__file__).parent / '.last_settings.pkl'
    
    if not settings_file.exists():
        return None
    
    try:
        with open(settings_file, 'rb') as f:
            settings = pickle.load(f)
        return settings
    except Exception as e:
        print(f"Warnung: Konnte Einstellungen nicht laden: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Video Dark Pixel Analyzer')
    parser.add_argument('video_path', nargs='?', default=None,
                        help='Pfad zur Video-Datei (.mp4) oder Dateiname im `Video`-Ordner')
    parser.add_argument('--output', '-o', default='analysis_results.json',
                        help='Ausgabedatei für Ergebnisse (default: analysis_results.json)')
    parser.add_argument('--threshold', '-t', type=int, default=50,
                        help='Intensitäts-Schwellenwert für dunkle Pixel (default: 50)')
    parser.add_argument('--list', action='store_true',
                        help='Zeige verfügbare Videos im `Video`-Ordner')
    parser.add_argument('--pick', action='store_true',
                        help='Interaktiv ein Video aus dem `Video`-Ordner auswählen')
    parser.add_argument('--recorded-fps', type=float, default=None,
                        help='(Optional) FPS, in der das Video aufgenommen wurde. Wird für Zeitberechnung verwendet.')
    
    args = parser.parse_args()
    
    # Ordner mit Beispiel-Videos (relativ zum Skript)
    video_dir = Path(__file__).parent / 'Video'

    # Versuche, letzte Einstellungen zu laden
    last_settings = None
    if not args.video_path and not args.pick:
        last_settings = load_last_settings()
    
    # Falls letzte Einstellungen vorhanden sind, frage ob sie verwendet werden sollen
    use_last_settings = False
    if last_settings:
        print("=== VIDEO DARK PIXEL ANALYZER ===")
        print("\nLetzte Einstellungen gefunden:")
        print(f"- Video: {last_settings.get('video_path', 'N/A')}")
        print(f"- FPS: {last_settings.get('recorded_fps', 'automatisch')}")
        print(f"- Threshold: {last_settings.get('threshold', 50)}")
        print(f"- Pendellänge: {last_settings.get('pendulum_length', 'keine Angabe')} m")
        print(f"- Hintergrund-Region: {last_settings.get('background_roi', 'N/A')}")
        print(f"- Dunkle Region: {last_settings.get('dark_roi', 'N/A')}")
        
        try:
            choice = input("\nMöchten Sie diese Einstellungen verwenden? (j/n): ").strip().lower()
            if choice in ['j', 'ja', 'y', 'yes']:
                use_last_settings = True
                args.video_path = last_settings.get('video_path')
                if last_settings.get('recorded_fps'):
                    args.recorded_fps = last_settings.get('recorded_fps')
                args.threshold = last_settings.get('threshold', 50)
        except EOFError:
            pass

    # Falls kein Video angegeben wurde und kein --pick/--list Flag gesetzt ist, interaktive Auswahl anbieten
    if args.video_path is None and not args.pick and not args.list and not use_last_settings:
        print("=== VIDEO DARK PIXEL ANALYZER ===")
        print("\nKein Video angegeben. Verfügbare Videos:")
        vids = find_videos_in_dir(video_dir)
        if not vids:
            print(f"Keine Video-Dateien im Ordner '{video_dir}' gefunden.")
            return
        for i, v in enumerate(vids):
            print(f"[{i}] {v.name}")
        
        try:
            sel = input('\nWähle Index oder Dateiname: ').strip()
        except EOFError:
            print("Keine Eingabe möglich.")
            return
        
        chosen = None
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(vids):
                chosen = vids[idx]
        else:
            # Suche nach exaktem Dateinamen
            for v in vids:
                if v.name == sel:
                    chosen = v
                    break
        
        if chosen is None:
            print('Ungültige Auswahl.')
            return
        args.video_path = str(chosen)

    # Falls ein Dateiname ohne Pfad angegeben wurde, prüfe im Video-Ordner
    if args.video_path is not None and not Path(args.video_path).exists():
        candidate = video_dir / args.video_path
        if candidate.exists():
            args.video_path = str(candidate)
        else:
            print(f"Fehler: Video-Datei '{args.video_path}' nicht gefunden.")
            return
    
    print("=== VIDEO DARK PIXEL ANALYZER ===")
    print(f"Analysiere Video: {args.video_path}")
    
    try:
        # Initialisiere Analyzer
        analyzer = VideoDarkPixelAnalyzer(args.video_path)
        
        # Lese ersten Frame für Region-Auswahl
        first_frame = analyzer.get_frame(0)
        if first_frame is None:
            print("Fehler: Kann Video nicht lesen.")
            return
        
        print(f"\nVideo-Information:")
        print(f"- FPS (Datei): {analyzer.fps}")
        print(f"- Frames: {analyzer.total_frames}")
        print(f"- Dauer: {analyzer.duration:.2f} Sekunden")
        print(f"- Auflösung: {analyzer.frame_width}x{analyzer.frame_height}")
        
        # Frage nach der aufgenommenen FPS (falls nicht per CLI gesetzt)
        if args.recorded_fps is None:
            if use_last_settings and last_settings.get('recorded_fps'):
                recorded_fps = last_settings.get('recorded_fps')
                print(f"Verwendet gespeicherte FPS: {recorded_fps}")
            else:
                try:
                    user_in = input(f"In welcher FPS wurde das Video aufgenommen? (Enter = automatisch erkannt {analyzer.fps:.2f}): ").strip()
                except EOFError:
                    user_in = ''
                if user_in:
                    try:
                        recorded_fps = float(user_in)
                    except ValueError:
                        print("Ungültige Eingabe für FPS, verwende die Dateifps.")
                        recorded_fps = analyzer.fps
                else:
                    recorded_fps = analyzer.fps
        else:
            recorded_fps = args.recorded_fps

        print(f"Verwendete FPS für Zeitberechnung: {recorded_fps}")

        # Setze used_recorded_fps im Analyzer, damit Zeitberechnung konsistent ist
        analyzer.used_recorded_fps = recorded_fps

        # Auswahl: ROI oder Pixel-basiert
        print("\n" + "="*50)
        print("AUSWAHL-MODUS:")
        print("[1] Region (ROI) - ganze rechteckige Bereiche")
        print("[2] Pixel - einzelne oder mehrere Pixel auswählen")
        
        analysis_mode = "roi"  # Standard
        if use_last_settings and 'analysis_mode' in last_settings:
            analysis_mode = last_settings.get('analysis_mode', 'roi')
            print(f"\nVerwendeter Modus aus Einstellungen: {analysis_mode}")
        else:
            try:
                mode_choice = input("\nWähle Modus [1/2] (Standard: 1): ").strip()
                if mode_choice == "2":
                    analysis_mode = "pixel"
                else:
                    analysis_mode = "roi"
            except EOFError:
                pass
        
        print(f"Gewählter Modus: {analysis_mode.upper()}")
        
        # Regionen/Pixel auswählen (oder aus letzten Einstellungen laden)
        print("\n" + "="*50)
        background_roi = None
        dark_roi = None
        background_pixels = None
        dark_pixels = None
        
        if analysis_mode == "roi":
            if use_last_settings and last_settings.get('background_roi') and last_settings.get('dark_roi'):
                background_roi = tuple(last_settings.get('background_roi'))
                dark_roi = tuple(last_settings.get('dark_roi'))
                print(f"Verwende gespeicherte Regionen:")
                print(f"- Hintergrund: {background_roi}")
                print(f"- Dunkle Region: {dark_roi}")
            else:
                # Hintergrund-Region auswählen
                background_roi = analyzer.select_region(first_frame, "Hintergrund-Region")
                
                # Dunkle Region auswählen
                print("\n" + "="*50)
                dark_roi = analyzer.select_region(first_frame, "dunkle Region")
            
            print(f"\nAusgewählte Regionen:")
            print(f"- Hintergrund: {background_roi}")
            print(f"- Dunkle Region: {dark_roi}")
        
        else:  # pixel mode
            if use_last_settings and last_settings.get('background_pixels') and last_settings.get('dark_pixels'):
                background_pixels = last_settings.get('background_pixels')
                dark_pixels = last_settings.get('dark_pixels')
                print(f"Verwende gespeicherte Pixel:")
                print(f"- Hintergrund: {len(background_pixels)} Pixel")
                print(f"- Dunkle Region: {len(dark_pixels)} Pixel")
            else:
                # Hintergrund-Pixel auswählen
                background_pixels = analyzer.select_pixels(first_frame, "Hintergrund-Pixel")
                
                # Dunkle Pixel auswählen
                print("\n" + "="*50)
                dark_pixels = analyzer.select_pixels(first_frame, "dunkle Region Pixel")
            
            print(f"\nAusgewählte Pixel:")
            print(f"- Hintergrund: {len(background_pixels)} Pixel")
            print(f"- Dunkle Region: {len(dark_pixels)} Pixel")
        
        print(f"- Intensitäts-Schwellenwert: {args.threshold}")
        
        # Analyse durchführen
        print("\n" + "="*50)
        analyzer.analyze_video(background_roi, dark_roi, args.threshold,
                             background_pixels=background_pixels, dark_pixels=dark_pixels)
        
        # Ergebnisse ausgeben
        analyzer.print_summary()
        
        # Ergebnisse speichern
        analyzer.save_results(args.output)
        analyzer.save_double_period_differences()
        
        # Speichere die verwendeten Einstellungen für nächstes Mal
        settings = {
            'video_path': str(args.video_path),
            'recorded_fps': recorded_fps,
            'threshold': args.threshold,
            'pendulum_length': pendulum_length,
            'analysis_mode': analysis_mode,
            'background_roi': background_roi,
            'dark_roi': dark_roi,
            'background_pixels': background_pixels,
            'dark_pixels': dark_pixels
        }
        save_last_settings(settings)
        
    except Exception as e:
        print(f"Fehler bei der Analyse: {e}")
        # Versuche, die Zeitdifferenzen trotz Fehler zu speichern, falls Ergebnisse vorhanden sind
        if 'analyzer' in locals() and analyzer.results:
            try:
                analyzer.save_double_period_differences()
            except Exception as save_err:
                print(f"Fehler beim Speichern der Ergebnisse: {save_err}")
    finally:
        if 'analyzer' in locals():
            analyzer.close()

if __name__ == "__main__":
    main()