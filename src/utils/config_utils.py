"""
Modul: config_utils.py
Teil von: GeoAI_Framework
Hilfsfunktionen für die Konfigurations-Verarbeitung.
"""

def get_task_mode(config):
    """
    Unified helper to determine the task mode (binary, regression, multiclass) 
    from either training.task_type or data.mask_type.
    """
    if not config:
        return 'binary'
        
    # 1. Bevorzugt: training.task_type (wird von app.py gesetzt)
    task_type = config.get('training', {}).get('task_type')
    if task_type:
        # Falls in app.py 'classification' ausgewählt wurde, mappen wir es auf 'multiclass' 
        # oder lassen es so, falls der Rest des Backends 'classification' versteht.
        if task_type == 'classification':
            # Check if it's actually multiclass or binary
            class_map = config.get('data', {}).get('class_map')
            if class_map and len(class_map) > 2:
                return 'multiclass'
            return 'binary'
        return task_type
        
    # 2. Fallback: data.mask_type (alte Configs oder Backend-Default)
    mask_type = config.get('data', {}).get('mask_type')
    if mask_type:
        return mask_type
        
    return 'binary' # Default fallback
