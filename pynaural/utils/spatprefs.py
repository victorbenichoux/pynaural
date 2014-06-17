import os
import inspect
from ..utils import debugtools as db

## Do not modify this file.
# To customize your user preferences, refer to ./localprefs.py

try:
    from ..utils import localprefs
    db.log_debug('Loaded preference file')
except ImportError:
    db.log_debug('No preference file found, a new one was created')
    utilspath = os.path.dirname(inspect.getfile(db))
    f = open(os.path.join(utilspath, 'localprefs.py'),'w')
    f.write("""## User Preferences\n# This file is used to store variables that are used in the other modules""")
    f.close()
    from ..utils import localprefs
    db.log_debug('Loaded preference file')


def get_pref(name, default = None):
    '''
    Returns the value of the preference ``name'', as defined in the localprefs.py script.
    
    If the variable dummy_pref is defined in the localprefs .py script as follows:
    dummy_pref = range(10)
    Then prefs.get_pref('dummy_pref') will return range(10)
    
    Additionally, a ``default'' kwd arg is provided so that the value of ``default'' is returned if the preference is not found.
    '''
    if hasattr(localprefs, name):
        return localprefs.__dict__[name]
    else:
        if not default is None:
            return default
        else:
            raise AttributeError('No user preference specified for key '+name+"\nMaybe you should create it in utils/localprefs.py ?")

def list_prefs():
    '''
    Returns the preferences variables' names, that is all the names of the variables defined in the utils/localprefs.py script.
    '''
    return localprefs.__dict__.keys()

def set_pref(name, value):
    localprefs.__dict__[name] = value
