#!/usr/bin/env python3
"""
Video Dark Pixel Analyzer
Erkennt dunkle Pixel in Videos im Vergleich zum Hintergrund und berechnet Differenzen.
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
import os

def setup_matplotlib():
    """Setup matplotlib for proper rendering"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

class VideoDarkPixelAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Video-Datei konnte nicht geöffnet werden: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialisierung der Variablen
        self.results = []
        self.darkest_pixel_values = []

    def select_region(self, frame, region_name, window_name="Region Selection"):
        """
        Ermöglicht die Auswahl einer Region im Frame
        """
        print(f"\nBitte wählen Sie die {region_name} aus:")
        print("1. Klicken Sie links oben auf die Region")
        print("2. Ziehen Sie eine rechteckige Auswahl")
        print("3. Drücken Sie Enter oder SPACE um zu bestätigen")

        display_frame = frame.copy()

        roi = cv2.selectROI(window_name, display_frame, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi[2] > 0 and roi[3] > 0:
            return roi
        else:
            raise ValueError(f"Keine gültige {region_name} ausgewählt")

    def extract_region_intensity(self, frame, roi):
        """
        Extrahiert die mittlere Intensität einer Region.
        """
        x, y, w, h = roi
        region = frame[y:y+h, x:x+w]

        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region

        return np.mean(gray_region)

    def analyze_video(self, background_roi, dark_roi, intensity_threshold=50):
        """
        Analysiert das Video, erkennt dunkle Pixel und speichert die dunkelsten Pixelwerte.
        """
        print(f"\nAnalysiere Video mit {self.total_frames} Frames...")
        print(f"Modus: ROI-basierte Analyse")
        print(f"- Hintergrund-ROI: {background_roi}")
        print(f"- Dunkle ROI: {dark_roi}")

        frame_count = 0

        try:
            total = max(1, int(self.total_frames))
        except Exception:
            total = 1
        update_interval = max(1, total // 100)

        self.darkest_pixel_values = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_val = np.min(gray_frame)
            normalized_min_val = 1.0 - (min_val / 255.0)
            self.darkest_pixel_values.append(normalized_min_val)

            frame_count += 1

            if frame_count % update_interval == 0 or frame_count == total:
                self._print_progress(frame_count, total)

        return self.darkest_pixel_values

    def _print_progress(self, current, total, bar_length=40):
        """Terminal-Progressbar"""
        try:
            percent = float(current) / float(total)
        except ZeroDivisionError:
            percent = 0.0
        filled = int(round(bar_length * percent))
        bar = '#' * filled + '-' * (bar_length - filled)

        print(f"\rProgress: |{bar}| {percent*100:6.2f}% ({current}/{total})", end='', flush=True)
        if current >= total:
            print()

    def get_frame(self, frame_number):
        """
        Liest einen spezifischen Frame
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame if ret else None

    def find_continuous_groups(self, values, threshold=0.65):
        """Findet Gruppen von aufeinanderfolgenden Werten über dem Schwellenwert und berechnet deren mittlere Position"""
        groups = []
        current_group = []

        for i, value in enumerate(values):
            if value > threshold:
                current_group.append(i + 1)
            else:
                if current_group:
                    avg_position = sum(current_group) / len(current_group)
                    groups.append(avg_position)
                    current_group = []

        if current_group:
            avg_position = sum(current_group) / len(current_group)
            groups.append(avg_position)

        return groups

    def calculate_position_differences(self, positions):
        """Berechnet Differenzen zwischen jeder Position und der übernächsten Position"""
        differences = []

        for i in range(len(positions) - 2):
            current_pos = positions[i]
            overnext_pos = positions[i + 2]
            diff = overnext_pos - current_pos
            differences.append(diff)

        return differences

    def save_differences(self, differences, output_path):
        """Speichert die Differenzen in eine Textdatei"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as file:
                for diff in differences:
                    file.write(f"{diff:.1f}\n".replace('.', ','))

            print(f"\nErgebnisse in '{output_path}' gespeichert")

        except Exception as e:
            print(f"\nFehler beim Speichern: {e}")

    def close(self):
        """
        Schließt die Video-Capture-Verbindung
        """
        self.cap.release()
        cv2.destroyAllWindows()

def find_videos_in_dir(video_dir):
    """
    Liefert eine Liste von Video-Dateien im angegebenen Ordner zurück.
    """
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    p = Path(video_dir)
    if not p.exists() or not p.is_dir():
        return []
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]
    return files

def main():
    parser = argparse.ArgumentParser(description='Video Dark Pixel Analyzer')
    parser.add_argument('video_path', nargs='?', default=None,
                        help='Pfad zur Video-Datei (.mp4) oder Dateiname im `Video`-Ordner')
    parser.add_argument('--threshold', '-t', type=int, default=50,
                        help='Intensitäts-Schwellenwert für dunkle Pixel (default: 50)')

    args = parser.parse_args()

    video_dir = Path(__file__).parent / 'Video'

    if args.video_path is None:
        print("=== Video Dark Pixel Analyzer ===")
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
            print("\nKeine Eingabe möglich.")
            return

        chosen = None
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(vids):
                chosen = vids[idx]
        else:
            for v in vids:
                if v.name == sel:
                    chosen = v
                    break

        if chosen is None:
            print('Ungültige Auswahl.')
            return
        args.video_path = str(chosen)

    if not Path(args.video_path).exists():
        candidate = video_dir / args.video_path
        if candidate.exists():
            args.video_path = str(candidate)
        else:
            print(f"Fehler: Video-Datei '{args.video_path}' nicht gefunden.")
            return

    print("=== Video Dark Pixel Analyzer ===")
    print(f"Analysiere Video: {args.video_path}")

    analyzer = None
    try:
        analyzer = VideoDarkPixelAnalyzer(args.video_path)
        first_frame = analyzer.get_frame(0)
        if first_frame is None:
            print("Fehler: Kann Video nicht lesen.")
            return

        print(f"\nVideo-Information:")
        print(f"- FPS (Datei): {analyzer.fps:.2f}")
        print(f"- Frames: {analyzer.total_frames}")
        print(f"- Dauer: {analyzer.duration:.2f} Sekunden")
        print(f"- Auflösung: {analyzer.frame_width}x{analyzer.frame_height}")

        print("\n" + "="*50)
        print("AUSWAHL-MODUS: ROI")

        background_roi = analyzer.select_region(first_frame, "Hintergrund-Region")
        print("\n" + "="*50)
        dark_roi = analyzer.select_region(first_frame, "dunkle Region")

        print(f"\nAusgewählte Regionen:")
        print(f"- Hintergrund: {background_roi}")
        print(f"- Dunkle Region: {dark_roi}")
        print(f"- Intensitäts-Schwellenwert: {args.threshold}")

        print("\n" + "="*50)

        darkest_pixels = analyzer.analyze_video(background_roi, dark_roi, args.threshold)

        if darkest_pixels:
            positions = analyzer.find_continuous_groups(darkest_pixels, threshold=0.8)
            differences = analyzer.calculate_position_differences(positions)

            out_dir = Path(__file__).parent / 'Ergebnisse'
            output_file = out_dir / "differenzen.txt"
            analyzer.save_differences(differences, output_file)
        else:
            print("\nKeine Daten zum Analysieren der Differenzen gefunden.")

    except Exception as e:
        print(f"\nFehler bei der Analyse: {e}")
    finally:
        if analyzer:
            analyzer.close()

if __name__ == "__main__":
    main()
