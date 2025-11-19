import os

def read_values(filename):
    """Liest Werte aus einer Textdatei und konvertiert sie zu float"""
    values = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    line = line.replace(',', '.')
                    try:
                        value = float(line)
                        values.append(value)
                    except ValueError:
                        continue
        return values
    except FileNotFoundError:
        print(f"Fehler: Datei '{filename}' nicht gefunden")
        return None
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        return None

def find_continuous_groups(values, threshold=0.65):
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

def calculate_position_differences(positions):
    """Berechnet Differenzen zwischen jeder Position und der übernächsten Position"""
    differences = []
    
    for i in range(len(positions) - 2):
        current_pos = positions[i]
        overnext_pos = positions[i + 2]
        diff = overnext_pos - current_pos
        differences.append(diff)
    
    return differences

def save_differences(differences, output_path):
    """Speichert die Differenzen in eine Textdatei"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            for diff in differences:
                file.write(f"{diff:.1f}\n".replace('.', ','))
        
        print(f"Ergebnisse in '{output_path}' gespeichert")
        
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

def main():
    input_file = "input.txt"
    output_dir = "Ergebnisse"
    output_file = os.path.join(output_dir, "differenzen.txt")
    
    values = read_values(input_file)
    
    if values is None:
        return
    
    positions = find_continuous_groups(values, threshold=0.8)
    differences = calculate_position_differences(positions)
    
    save_differences(differences, output_file)

if __name__ == "__main__":
    main()