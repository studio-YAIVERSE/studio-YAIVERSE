"""
History
    - 230419 : MINSU init
        - get statistic of selected category's obj files
"""


import os
import glob
from tqdm import tqdm
import numpy as np

CATEGORIES = {
    #'table':'04379243',
    'car':'02958343',
    'chair':'03001627',
    # 'airplane':'02691156',
    # 'sofa':'04256520',
    # 'watercraft':'04530566',
    # 'bus':'02924116'
    'loudspeaker': '03691459',
    'telephone':'04401088',
    'cabinet':'02933112',
    # 'motorbike': '03790512',
}


if __name__ == "__main__":
    
    data_root_dir = '/opt/myspace/data/ShapeNetCore.v1'   # path for shapenet
    
    categories = ['car', 'chair', 'loudspeaker', 'telephone', 'cabinet']    # select keys 
    
    categories_info = {}
    
    for c in categories:
        categories_info[c] = {'x':[], 'y':[], 'z':[], 'size':[]}
        # 'axis' : [store info. of one mesh x N]
    
    for category in categories:
        dir_ = os.path.join(data_root_dir, CATEGORIES[category])
        
        sub_dir = sorted(glob.glob(dir_ +'/*'))
       
        for sub in tqdm(sub_dir):
            
            # inspect 3D bbox of one mesh
            
            vtx_x = []
            vtx_y = []
            vtx_z = []
            
            with open(sub +'/model.obj', 'r') as f:
                
                while True:
                    line = f.readline()
                    
                    if not line:
                        break
                    
                    if line[:2] == 'v ':
                        
                        coords = line.split(' ')
                        
                        vtx_x.append(float(coords[1]))
                        vtx_y.append(float(coords[2]))
                        vtx_z.append(float(coords[3]))
                        
            line_x = abs(min(vtx_x) - max(vtx_x))
            line_y = abs(min(vtx_y) - max(vtx_y))
            line_z = abs(min(vtx_z) - max(vtx_z))
            
            if line_x > 0 and line_y > 0 and line_z > 0:    # filter invalid
                categories_info[category]['x'].append(line_x)
                categories_info[category]['y'].append(line_y)
                categories_info[category]['z'].append(line_z)
                categories_info[category]['size'].append(line_x * line_y * line_z)
            
        categories_info[category]['x'] = [np.mean(categories_info[category]['x']), np.std(categories_info[category]['x']), np.max(categories_info[category]['x']), np.min(categories_info[category]['x'])]
        categories_info[category]['y'] = [np.mean(categories_info[category]['y']), np.std(categories_info[category]['y']), np.max(categories_info[category]['y']), np.min(categories_info[category]['y'])]
        categories_info[category]['z'] = [np.mean(categories_info[category]['z']), np.std(categories_info[category]['z']), np.max(categories_info[category]['z']), np.min(categories_info[category]['z'])]
        categories_info[category]['size'] = [np.mean(categories_info[category]['size']), np.std(categories_info[category]['size']), np.max(categories_info[category]['size']), np.min(categories_info[category]['size'])]

    # final result
    print(categories_info)
    
    import json

    with open('scale_info_0507.json', 'w') as f:
        json.dump(categories_info, f, indent=4)

                    
                    
                        
        