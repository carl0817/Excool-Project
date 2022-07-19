import numpy as np
import os
import tempfile

def ciftiopen(filename, caret7command):
    # Open a CIFTI file by converting to GIFTI external binary first and then
    # using the GIFTI toolbox

    # grot = fileparts(filename)
    with open(filename) as f:
        for line in f:
            drive, path = os.path.splitdrive(line)
            path, filename = os.path.split(path)
            print(f'Drive is {drive} Path is {path} and file is {filename}')

    
    # if size(grot,1)==0:
    if grot.shape()==0:
        grot='.'



    # tmpname = tempname
    tmpname = tempfile.TemporaryFile().name
    # unix([caret7command ' -cifti-convert -to-gifti-ext ' filename ' ' tmpname '.gii']);
    os.system(f'{caret7command} -cifti-convert -to-gifti-ext {filename} {tmpname}.gii')
    # cifti = gifti([tmpname '.gii']);
    cifti = gifti(f'{tmpname}.gii') # ??
    # unix(['rm ' tmpname '.gii ' tmpname '.gii.data']);
    os.system(f'rm {tmpname}.gii {tmpname}.gii.data')

    return cifti