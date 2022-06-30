import json, random
from pickle import OBJ
from os.path import join
from re import S
from tqdm import tqdm

import pandas as pd

from global_variables import *
from utils import *


corpus = pd.read_csv(join(DATA_PATH, COMMON_SENSE_CORPUS))

def attr_to_name(size, color, shape, material):
    name_candidates = corpus[
        (corpus['size'] == size)
        & (corpus['color'] == color)
        & (corpus['shape'] == shape)
        & (corpus['material'] == material)
    ]['object']
    name_candidates = list(set(name_candidates))
    assert len(name_candidates) < 3
    name = random.choice(name_candidates)
    return name


class SceneGraphGenerator:
    def __init__(self):
        with open(join(join(DATA_PATH, args.task), OBJECTS_FILE)) as f:
            self.objects = f.read().splitlines()
        with open(join(join(DATA_PATH, args.task), COLORS_FILE)) as f:
            self.colors = f.read().splitlines()
        
        self.opposite_reltion = {
            'left': 'right',
            'right': 'left',
            'front': 'behind',
            'behind': 'front'
        }

    def convert_CLEVR_to_GQA(self, filename=None) -> list: # list of output file names
        """ Convert CLEVR dataset to GQA-style """

        filenames = []
        for split in ['train', 'val']:
            input_filename = f"CLEVR_{split}_scenes.json"
            output_filename = self._convert_CLEVR_to_GQA(
                input_filename=input_filename,
                output_filename=filename
                )
            filenames.append(output_filename)
        return filenames

    def _convert_CLEVR_to_GQA(self, input_filename, output_filename=None) -> str: # output file name
        
        with open(join(DATA_PATH, input_filename)) as f:
            clevr_dataset = json.load(f)
            
        clevr_scenes = clevr_dataset['scenes']

        gqa_scenes = {}
        for clevr_scene in tqdm(clevr_scenes):
            gqa_scene = {}
            clevr_relations = clevr_scene['relationships']
            for obj_id, clevr_obj in enumerate(clevr_scene['objects']):
                # TODO: now we do not use attr
                size = clevr_obj['size']
                color = clevr_obj['color']
                # if color == "cyan": # OOV: cyan
                #     color = "white"
                shape = clevr_obj['shape']
                material = clevr_obj['material']
                name = attr_to_name(size, color, shape, material)

                left_objs = clevr_relations['left'][obj_id]
                right_objs = clevr_relations['right'][obj_id]
                front_objs = clevr_relations['front'][obj_id]
                behind_objs = clevr_relations['behind'][obj_id]

                gqa_relations = \
                    [ {'object': str(left_obj_id), 'name': 'left'}
                    for left_obj_id in left_objs ] + \
                    [ {'object': str(right_obj_id), 'name': 'right'}
                    for right_obj_id in right_objs ] + \
                    [ {'object': str(front_obj_id), 'name': 'front'}
                    for front_obj_id in front_objs ] + \
                    [ {'object': str(behind_obj_id), 'name': 'behind'}
                    for behind_obj_id in behind_objs ]
                
                x, y, z = clevr_obj['3d_coords']
                
                gqa_obj = {
                    'name': name,
                    'relations': gqa_relations,
                    'attributes': [], # [size, color, shape, material], # TODO: now we do not use attr
                    'x': x,
                    'y': y,
                    'z': z # new components
                }

                gqa_scene[obj_id] = gqa_obj
        
            gqa_scenes[clevr_scene['image_index']] = gqa_scene

        if output_filename is None:
            output_filename = input_filename.replace('CLEVR', 'my')
        with open(join(DATA_PATH, output_filename), 'w') as f:
            json.dump(gqa_scenes, f)

        logger.info(f"Save as {join(DATA_PATH, output_filename)}")
        return output_filename
    
    def generate(
        self,
        min_num_objects=3,
        max_num_objects=3,
        train_size=8000,
        val_size=2000
        ):

        output_filenames = []
        train_scene_graphs = {}
        for image_index in range(train_size):
            num_objects = random.randint(min_num_objects, max_num_objects)
            one_scene_graph = self.generate_one_scene_graph(num_objects)
            train_scene_graphs[str(image_index)] = one_scene_graph
        
        filename = join(join(DATA_PATH, args.task), 'my_train_scenes.json')
        with open(filename, 'w') as f:
            json.dump(train_scene_graphs, f)
        output_filenames.append('my_train_scenes.json')
        logger.info(f"Save as my_train_scenes.json")
        
        val_scene_graphs = {}
        for image_index in range(val_size):
            num_objects = random.randint(min_num_objects, max_num_objects)
            one_scene_graph = self.generate_one_scene_graph(num_objects)
            val_scene_graphs[str(image_index)] = one_scene_graph

        filename = join(join(DATA_PATH, args.task), 'my_val_scenes.json')
        with open(filename, 'w') as f:
            json.dump(val_scene_graphs, f)
        output_filenames.append('my_val_scenes.json')
        logger.info(f"Save as my_val_scenes.json")

        return output_filenames
    
    def generate_one_scene_graph(self, num_objects=3):
        scene_graph = {}
        random.shuffle(self.colors)
        for obj_id in range(num_objects):
            name = random.choice(self.objects)
            color = self.colors[obj_id % len(self.colors)] # TODO: more attributes
            x = random.random() * 2 - 1 # assume to be normalized \in (-1, 1)
            y = random.random() * 2 - 1
            z = random.random() * 2 - 1
            relations = []

            for past_obj_id in range(obj_id):
                past_obj = scene_graph[str(past_obj_id)]
                past_x = past_obj['x']
                past_y = past_obj['y']

                rel_x = 'left' if x < past_x else 'right'
                rel_y = 'front' if y < past_y else 'behind'

                relations.append(
                    {
                        'object': str(past_obj_id),
                        'name': rel_x
                    }
                )
                relations.append(
                    {
                        'object': str(past_obj_id),
                        'name': rel_y
                    }
                )

                # updates past object's relations
                past_relations = scene_graph[str(past_obj_id)]['relations']
                past_relations.append(
                    {
                        'object': str(obj_id),
                        'name': self.opposite_reltion[rel_x]
                    }
                )
                past_relations.append(
                    {
                        'object': str(obj_id),
                        'name': self.opposite_reltion[rel_y]
                    }
                )

            obj = {
                'name': name,
                'relations': relations,
                'attributes': ['', color, '', ''], # size, color, shape, material
                'x': x,
                'y': y,
                'z': z
            }
            scene_graph[str(obj_id)] = obj

        return scene_graph


class TextGenerator:
    def __init__(self, args, scene_graph_filename):
        self.args = args

        self.scene_graph_filename = scene_graph_filename
        self.opposite_reltion = {
            'left': 'right',
            'right': 'left',
            'front': 'behind',
            'behind': 'front'
        }
        self.ordinal_numbers = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']

        self.scenes = open_dataset(scene_graph_filename)

        self.name_template = open_template(NAME_TEMPLATE)
        self.attr_template = open_template(ATTR_TEMPLATE)
        self.single_rel_template = open_template(SIGNLE_REL_TEMPLATE)
        self.most_rel_template = open_template(MOST_REL_TEMPLATE)
        self.ordinal_rel_template = open_template(ORDINAL_REL_TEMPLATE) 
        self.common_sense_df = corpus
    
    def get_det(self, word):
        det = random.choice(['the', 'a'])
        # TODO: fix hard coding
        an_words = {
            'object',
            'airliner',
            'airship',
            'altar',
            'ant',
            'ear',
            'envelope',
            'ipod',
            'iron',
            'oboe',
            'organ',
            'orange',
            'umbrella'
        }
        if det == 'a':
            if word in an_words:
                det = 'an'
        return det
    

    def generate_text_dataset(
        self,
        num_name=0,
        num_attr=0,
        num_single_rel=0,
        num_most_rel=0,
        num_common_sense=0,
        num_ordinal_rel=0,
        filename=None
        ) -> str: # file name
        """ Generate text dataset """

        if num_name + num_attr + num_single_rel + num_most_rel \
            + num_common_sense + num_ordinal_rel == 0:
            num_name = int(args.gen_text[0])
            num_attr = int(args.gen_text[1])
            num_single_rel = int(args.gen_text[2])
            # num_double_rel = int(args.gen_text[3])
            num_most_rel = int(args.gen_text[4])
            num_common_sense = int(args.gen_text[5])
            num_ordinal_rel = int(args.gen_text[6])

        logger.debug(f"{num_name} name")
        logger.debug(f"{num_attr} attribute")
        logger.debug(f"{num_single_rel} single relatoin")
        # logger.debug(f"{num_double_rel} double relatoin")
        logger.debug(f"{num_most_rel} most relatoin")
        logger.debug(f"{num_common_sense} common sense")
        logger.debug(f"{num_ordinal_rel} ordinal relatoin")

        text_dataset = {}
        for idx, scene in tqdm(self.scenes.items()):
            self.idx = idx
            text_dataset[idx] = {
                'number of labels': len(scene),
                'data': []
            }

            for _ in range(num_name):
                text, label = self.generate_name_txt(scene)
                text_dataset[idx]['data'].append(
                    {'text': text, 'label': label}
                )
            for _ in range(num_attr):
                text, label = self.generate_attr_text(scene)
                text_dataset[idx]['data'].append(
                    {'text': text, 'label': label}
                )
            for _ in range(num_single_rel):
                text, label = self.generate_single_rel_text(scene)
                text_dataset[idx]['data'].append(
                    {'text': text, 'label': label}
                )
            for _ in range(num_most_rel):
                text, label = self.generate_most_rel_text(scene)
                text_dataset[idx]['data'].append(
                    {'text': text, 'label': label}
                )
            for _ in range(num_common_sense):
                text, label = self.generate_common_sense_text(scene)
                text_dataset[idx]['data'].append(
                    {'text': text, 'label': label}
                )
            for _ in range(num_ordinal_rel):
                text, label = self.generate_ordinal_rel_text(scene)
                text_dataset[idx]['data'].append(
                    {'text': text, 'label': label}
                )

        if filename is None:
            filename = self.scene_graph_filename.replace("scenes", "texts").replace(".json", '_')
            filename += self.args.gen_text + '.json'
        with open(join(join(DATA_PATH, self.args.task), filename), 'w') as f:
            json.dump(text_dataset, f)
        
        logger.info(f"Save as {join(join(DATA_PATH, self.args.task), filename)}")
        return filename


    def generate_name_txt(self, scene: dict) -> tuple: # (text, label)
        """ Generate text with name and corresponding label """
        
        obj_names = [obj['name'] for obj in scene.values()]

        word = random.choice(obj_names)
        text = random.choice(self.name_template['text_list'])
        text = text.replace('<N>', word)
        text = text.replace('<D>', self.get_det(word))
        text = ' '.join(text.split())

        return text, word
    
    def generate_attr_text(self, scene):
        obj_ids = list(scene.keys())
        obj_id = random.choice(obj_ids)
        obj = scene[obj_id]
        name = obj['name']
        size, color, shape, material = obj['attributes']
        
        text = random.choice(self.attr_template['text_list'])
        text = text.replace('<N>', name)
        text = text.replace('<D>', self.get_det(name))
        text = text.replace('<C>', color)
        text = ' '.join(text.split())

        if self.args.label_type == 'name':
            label_type = name
        elif self.args.label_type == 'color':
            label_type = color
        else:
            raise NotImplementedError()

        return text, label_type

    def generate_single_rel_text(self, scene):

        # single label classification
        target_rel_name = random.choice(['left', 'right', 'front', 'behind'])
        opposite_rel_name = self.opposite_reltion[target_rel_name]
        scene_items = list(scene.items())
        random.shuffle(scene_items)
        target_obj_id = -1
        for obj_id, obj in scene_items:
            relations = obj['relations']
            target_relations = [rel for rel in relations if rel['name'] == target_rel_name]
            if len(target_relations) == 1:
                target_obj_id = obj_id
                rel_obj_id = target_relations[0]['object']
                break

        assert target_obj_id != -1
        target_obj_name = scene[target_obj_id]['name']
        rel_obj_name = scene[rel_obj_id]['name']

        text = random.choice(self.single_rel_template['texts'])
        things = ['object', 'thing']
        thing = random.choice(things)
        text = text.replace('<O1>', thing)
        text = text.replace('<D1>', self.get_det(thing))
        text = text.replace("<R>", opposite_rel_name)

        if '<N>' in text:
            text = text.replace('<N>', target_obj_name)
            text = text.replace('<D2>', self.get_det(target_obj_name))
        else: # TODO: We do not use any attributes
            raise Exception("We do not use any attributes")
        
        return text, rel_obj_name

        # multi label classification?
        target_obj_id = random.choice(list(scene.keys()))
        target_obj = scene[target_obj_id]
        rel_obj = random.choice(target_obj['relations'])
        rel_obj_id = rel_obj['object']
        rel_obj_name = scene[rel_obj_id]['name']
        rel_name = rel_obj['name']
        opposite_rel_name = self.opposite_reltion[rel_name]

        text = random.choice(self.one_rel_template['texts'])

        things = ['object', 'thing']
        thing = random.choice(self.things)
        text = text.replace('<O1>', thing)
        text = text.replace('<D1>', self.get_det(thing))
        text = text.replace("<R>", opposite_rel_name)

        if '<N>' in text:
            text = text.replace('<N>', rel_obj_name)
            text = text.replace('<D2>', self.get_det(rel_obj_name))
        else: # TODO: We do not use any attributes
            raise Exception("We do not use any attributes")
            # size, color, _, _ = scene[rel_obj_id]['attributes']
            # text = text.replace('<Z>', size)
            # text = text.replace('<C>', color)
            # thing = random.choice(things)
            # text = text.replace('<O2>', thing)
            # text = text.replace('<D2>', self.get_det(rel_obj_name))

        text = ' '.join(text.split())
        
        return text, rel_obj_name    
    
    def generate_most_rel_text(self, scene):
        target_rel = random.choice(["left", "right", "front", "behind"])

        # DFS: https://data-marketing-bk.tistory.com/44
        need_visited, visited = [], []
        start_node = random.choice(list(scene.keys()))
        need_visited.append(start_node)
        target_id = -1
        while need_visited:
            node = need_visited.pop()
            if node not in visited:
                visited.append(node)
                
                next_nodes = [
                    obj['object'] 
                    for obj
                    in scene[node]['relations'] 
                    if obj['name'] == target_rel
                ]
                if not next_nodes:
                    target_id = node
                    break
                need_visited.extend(next_nodes)

        assert target_id != -1
        name = scene[target_id]['name']
        size, color, shape, material = scene[target_id]['attributes']
        opposite_target_rel = self.opposite_reltion[target_rel]

        text = random.choice(self.most_rel_template['texts'])
        text = text.replace('<R>', opposite_target_rel)

        things = ['object', 'thing']
        text = text.replace('<O>', random.choice(things))
        text = ' '.join(text.split())
        
        if self.args.label_type == 'name':
            label_type = name
        elif self.args.label_type == 'color':
            label_type = color
        else:
            raise NotImplementedError()

        return text, label_type
    
    def generate_common_sense_text(self, scene):
        target_obj_id = random.choice(list(scene.keys()))
        target_obj = scene[target_obj_id]
        target_obj_name = target_obj['name']

        df = self.common_sense_df
        text_candidates = df[df['object'] == target_obj_name]['text']
        text_candidates = list(set(text_candidates))
        text = random.choice(text_candidates)

        return text, target_obj_name
    
    def generate_ordinal_rel_text(self, scene):
        target_obj_id = random.choice(list(scene.keys()))
        target_obj = scene[target_obj_id]
        target_obj_name = target_obj['name']

        rel_names = ['left', 'right', 'front', 'behind']
        target_rel_name = random.choice(rel_names)

        opposite_reltion = {
            'left': 'right',
            'right': 'left',
            'front': 'behind',
            'behind': 'front'
        }
        opposite_rel_name = opposite_reltion[target_rel_name]
        
        target_count = 0
        opposite_count = 0
        for rel in target_obj['relations']:
            if rel['name'] == target_rel_name:
                target_count += 1
            elif rel['name'] == opposite_rel_name:
                opposite_count += 1
        
        assert target_count + opposite_count + 1 == len(scene), \
            f"{target_count} + {opposite_count} + 1 != {len(scene)}"
        ordinal_number = self.ordinal_numbers[opposite_count + 1]
        assert ordinal_number

        text = random.choice(self.ordinal_template['texts'])
        things = ['object', 'thing', 'one', 'item']
        thing = random.choice(things)
        text = text.replace('<Or>', ordinal_number)
        text = text.replace('<O>', thing)
        if target_rel_name == 'behind':
            target_rel_name = 'back'
        text = text.replace('<R>', target_rel_name)

        return text, target_obj_name



if __name__ == '__main__':
    import os
    from arguments import get_args

    args = get_args()
    seed_all()

    logger.info(f"Generate dataset for {args.task}")

    scene_graph_filenames = []
    if args.gen_scene:
        logger.info("Generate scene graph dataset")
        scene_graph_generator = SceneGraphGenerator()
        if args.task == 'manip':
            scene_graph_filenames = scene_graph_generator.convert_CLEVR_to_GQA()
        elif args.task == 'nav':
            scene_graph_filenames = scene_graph_generator.generate()


    if args.gen_text != '0000000':
        if not scene_graph_filenames:
            for split in ['train', 'val']:
                if os.path.isfile(join(join(DATA_PATH, args.task), f"my_{split}_scenes.json")):
                    scene_graph_filenames.append(f"my_{split}_scenes.json")
        if not scene_graph_filenames:
            raise Exception("No scene graph dataset")
            
        logger.info("Generate text dataset")

        for scene_graph_filename in scene_graph_filenames:
            text_generator = TextGenerator(args, scene_graph_filename)
            text_generator.generate_text_dataset()
    
    logger.info("done")
