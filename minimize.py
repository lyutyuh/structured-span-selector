from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import tempfile
import subprocess
import collections
from nltk import Tree

import conll
from transformers import AutoTokenizer

import copy
import nltk

SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'
GENRE_DICT = {"bc":"[unused20]", "bn":"[unused21]", "mz":"[unused22]", "nw":"[unused23]", "pt":"[unused24]", "tc":"[unused25]", "wb":"[unused26]"}

def recur_list_tree(parse_tree, offset=0):
    span_to_tag = {}
    for node in parse_tree:
        if type(node) == type(parse_tree):
            span_to_tag.update(recur_list_tree(node, offset))
            n_leaves = len(node.leaves())
            start, end = offset, offset+n_leaves-1
            span_to_tag[(start,end)] = node.label()
            offset += n_leaves
        else:
            offset += 1
            pass
    return span_to_tag

def get_chunks_from_tree(parse_tree,chunk_types,offset=0,depth=0):
    chunks = {}
    root = parse_tree
    for node in parse_tree:
        if type(node) == type(parse_tree):
            n_leaves = len(node.leaves())
            start, end = offset, offset+n_leaves-1
            if node.label() not in chunk_types:
                chunks.update(get_chunks_from_tree(node,chunk_types, offset, depth))
            else:
                if depth == 0:
                    chunks[(start,end)] = node.label()
                else:
                    next_level = get_chunks_from_tree(node,chunk_types, offset, depth-1)
                    if len(next_level) > 0:
                        chunks.update(next_level)
                    else:
                        chunks[(start,end)] = node.label()
            offset += n_leaves
        else: # leaf node
            offset += 1
            pass
    return chunks

def keep_coreferable(root):
    coreferable_types = ["NP", "VB", "PRP", "NML","CD","NNP"]
    processed_types = ["MNP", "V", "PRP", "NML", "NNP"]
    result_tree = copy.deepcopy(root)
    
    coreferable_spans = {}
             
    def recur(node, parent, parent_span, span, index_in_siblings):
        child_left_ind = span[0]
        res = []
        if True:
            for (i, child) in enumerate(node):
                if type(child) == nltk.tree.Tree:
                    child_span = (child_left_ind, child_left_ind+len(child.leaves()))
                    child_left_ind += len(child.leaves())
                    processed_node = recur(child, node, span, child_span, i)

                    res += processed_node
                else:
                    if node.label() not in ["CD", "NML","NNP","PRP"]: 
                        res += [node]
                    else:
                        res += [child]
        if node.label() == "NP":
            if True: # keeping all NPs parent.label() not in ["NP",]:
                node.set_label("MNP")
            else:
                if node[-1].label() in {"POS"}:
                    node.set_label("MNP")
                    pass
                elif any([x.label() in ("VP", "CC","HYPH") for x in parent]):
                    node.set_label("MNP")
                    pass
                elif sum([(x.label() in ("NP", "MNP")) for x in parent]) > 1:
                    node.set_label("MNP")
                    pass
                    
        elif node.label().startswith("PRP"):
            node.set_label("PRP") 
        elif node.label() == "CD":
            # exclude a lot of CDs
            if index_in_siblings > 0 and parent[index_in_siblings-1].label() == "DT":
                pass
            elif index_in_siblings < len(parent)-1 and parent[index_in_siblings+1].label() in ("NN", "NNS", ):
                pass
            else:
                node.set_label("-CD")
        elif node.label() in {"NN", "NNP"}:
            # exclude a lot of NNPs
            if index_in_siblings < len(parent)-1 and parent[index_in_siblings+1].label() == "POS":
                # node.set_label("-NNP") 
                pass
            elif any([(x.label() in ("CC", "HYPH")) for x in parent]):
                node.set_label("NNP") 
                pass
            elif index_in_siblings > 0 and parent[index_in_siblings-1].label() == "DT":
                pass
            elif index_in_siblings < len(parent)-1 and parent[index_in_siblings+1].label() in ("NN", "NNS", ):
                pass
            elif index_in_siblings == 0:
                pass
            else:
                node.set_label("-NNP")
                pass
        elif node.label().startswith("VB"):
            node.set_label("-V")
            pass
        if node.label() in processed_types:
            coreferable_spans[(span[0], span[1]-1)] = node.label()
            res = [nltk.Tree(node.label(), res)]
        else:
            res = res        
        return res
    result_tree = recur(result_tree, None, None, (0, len(result_tree.leaves())), 0)
    result_tree = nltk.Tree("TOP", result_tree)
    return result_tree, coreferable_spans

class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.speakers = []
        self.segment_info = []

        self.chunk_tags = [] # corresponding to subtokens
        self.constituents = []
        self.coreferables = []
        self.pos_tags = []
        

    def finalize(self):
        # finalized: segments, segment_subtoken_map
        # populate speakers from info
        subtoken_idx = 0
        for segment in self.segment_info:
            speakers = []
            for i, tok_info in enumerate(segment):
                if tok_info is None and (i == 0 or i == len(segment) - 1):
                    speakers.append('[SPL]')
                elif tok_info is None:
                    speakers.append(speakers[-1])
                else:
                    speakers.append(tok_info[9])
                    if tok_info[4] == 'PRP':
                        self.pronouns.append(subtoken_idx)
                subtoken_idx += 1
            self.speakers += [speakers]
        # populate sentence map

        # populate clusters
        first_subtoken_index = -1
        for seg_idx, segment in enumerate(self.segment_info):
            speakers = []
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                coref = tok_info[-2] if tok_info is not None else '-'
                if coref != "-":
                    last_subtoken_index = first_subtoken_index + tok_info[-1] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(first_subtoken_index)
                        else:
                            cluster_id = int(part[:-1])
                            start = self.coref_stacks[cluster_id].pop()
                            self.clusters[cluster_id].append((start, last_subtoken_index))
        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        
        flattened_sentences = flatten(self.segments)
        all_mentions = flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        
        
        for cluster in merged_clusters:
            for mention in cluster:
                if subtoken_map[mention[0]] == subtoken_map[mention[1]]:
                    if self.pos_tags[subtoken_map[mention[0]]].startswith("V"):
                        self.coreferables.append(mention)
                        
                        
        assert len(all_mentions) == len(set(all_mentions))
        chunk_tags = flatten(self.chunk_tags)
        num_words =  len(flattened_sentences)
        assert num_words == len(flatten(self.speakers))
        # assert num_words == len(chunk_tags), (num_words, len(chunk_tags))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        def mapper(x):
            if x == "NP":
                return 1
            else:
                return 2
        return {
          "doc_key": self.doc_key,
          "sentences": self.segments,
          "speakers": self.speakers,
          "constituents": [x[0] for x in self.constituents], # 
          "constituent_type": [x[1] for x in self.constituents], #         
          # "coreferables": self.coreferables,
          "ner": [],
          "clusters": merged_clusters,
          'sentence_map':sentence_map,
          "subtoken_map": subtoken_map,
          'pronouns': self.pronouns,
          "chunk_tags": self.chunk_tags
        }


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    previous_token = 0
    final_chunk_tags = []
    index_mapping_dict = {}
    cur_seg_ind = 1
    while current < len(document_state.subtokens):
        # -3 for 3 additional special tokens
        end = min(current + max_segment_len - 1 - 3, len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(current + max_segment_len - 1 - 3, len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")

        for i in range(current, end+1):
            index_mapping_dict[i] = i + 3*cur_seg_ind - 1
            
        
        genre = document_state.doc_key[:2]
        genre_text = GENRE_DICT[genre]
        document_state.tokens.append(genre_text)
            
        document_state.segments.append(['[CLS]', genre_text] + document_state.subtokens[current:end + 1] + ['[SEP]'])

        subtoken_map = document_state.subtoken_map[current : end + 1]
        document_state.segment_subtoken_map.append([previous_token, previous_token] + subtoken_map + [subtoken_map[-1]])
        info = document_state.info[current : end + 1]
        document_state.segment_info.append([None, None] + info + [None])
        current = end + 1
        cur_seg_ind += 1
        previous_token = subtoken_map[-1]

    document_state.chunk_tags = final_chunk_tags
    return index_mapping_dict

def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s)-3 for s in segments])
    for segment in segments:
        sent_map.append(current)
        sent_map.append(current)
        for i in range(len(segment) - 3):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
        sent_map.append(current)
    return sent_map

def get_document(document_lines, tokenizer, language, segment_len):
    document_state = DocumentState(document_lines[0])
    word_idx = -1
    parse_pieces = []
    cur_sent_offset = 0
    cur_sent_len = 0
    
    current_speaker = None
    
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            
            if current_speaker is None or current_speaker != row[9]:
                added_speaker_head = True
                # insert speaker
                word_idx += 1
                current_speaker = row[9]
                speaker_text = tokenizer.tokenize(current_speaker)
                parse_pieces.append(f"(PSEUDO {' '.join([SPEAKER_START] + speaker_text + [SPEAKER_END])}")
                document_state.tokens.append(current_speaker)
                document_state.pos_tags.append("SPEAKER")
                for sidx, subtoken in enumerate([SPEAKER_START] + speaker_text + [SPEAKER_END]):
                    cur_sent_len += 1
                    document_state.subtokens.append(subtoken)
                    info = None 
                    document_state.info.append(info)
                    document_state.sentence_end.append(False)
                    document_state.subtoken_map.append(word_idx)
                
            word_idx += 1
            word = normalize_word(row[3], language)

            parse_piece = row[5]
            pos_tag = row[4]
            if pos_tag == "(":
                pos_tag = "-LRB-"
            if pos_tag == ")":
                pos_tag = "-RRB-"

            (left_brackets, right_hand_side) = parse_piece.split("*")
            right_brackets = right_hand_side.count(")") * ")"

            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.pos_tags.append(pos_tag)
            
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                cur_sent_len += 1
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
            new_core = " ".join(subtokens)
            
            parse_piece = f"{left_brackets} {new_core} {right_brackets}"
            # parse_piece = f"{left_brackets} {new_core} {right_brackets}"
            parse_pieces.append(parse_piece)
        else:
            if added_speaker_head:
                parse_pieces.append(")")
                added_speaker_head = False
            
            parse_tree = Tree.fromstring("".join(parse_pieces))
            chunk_dict = get_chunks_from_tree(parse_tree, ["NP"])
            constituent_dict = recur_list_tree(parse_tree)
            
            coreferable_spans_dict = keep_coreferable(parse_tree)[1]

            document_state.coreferables += [[x[0]+cur_sent_offset,x[1]+cur_sent_offset] for x,y in coreferable_spans_dict.items()]
            document_state.constituents += [[[x[0]+cur_sent_offset,x[1]+cur_sent_offset],y] for x,y in constituent_dict.items()]
            sent_chunk_tags = ["O" for i in range(cur_sent_len)]
            if pos_tag==".":
                sent_chunk_tags[-1] = "."
            for (chunk_l, chunk_r), chunk_tag in chunk_dict.items():
                chunk_len = chunk_r - chunk_l + 1
                if chunk_len == 1:
                    sent_chunk_tags[chunk_r] = "U-" + chunk_tag
                else:
                    sent_chunk_tags[chunk_l] = "B-" + chunk_tag
                    sent_chunk_tags[chunk_r] = "L-" + chunk_tag
                    for i in range(chunk_l + 1, chunk_r):
                        sent_chunk_tags[i] = "I-" + chunk_tag
            document_state.sentence_end[-1] = True
            cur_sent_offset += cur_sent_len
            cur_sent_len, parse_pieces = 0, []
            document_state.chunk_tags += sent_chunk_tags

    constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    index_mapping_dict = split_into_segments(document_state, segment_len, constraints1, document_state.token_end)

    for x in document_state.constituents:
        x[0][0] = index_mapping_dict[x[0][0]]
        x[0][1] = index_mapping_dict[x[0][1]]
    for x in document_state.coreferables:
        x[0] = index_mapping_dict[x[0]]
        x[1] = index_mapping_dict[x[1]]
    
    stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
    document = document_state.finalize()
    return document

def skip(doc_key):
    # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
        # return True
    return False

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
    input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
    count = 0
    print("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    num_coreferables, num_words = 0, 0
    with open(output_path, "w") as output_file:
        for document_lines in documents:
            if skip(document_lines[0]):
                continue
            document = get_document(document_lines, tokenizer, language, seg_len)
            
            # num_coreferables += len(document["coreferables"])
            num_words += len(flatten(document["sentences"]))
            
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}, with {} words".format(count, output_path, num_words))

def minimize_language(language, labels, stats, seg_len, input_dir, output_dir, do_lower_case):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096")
    
    minimize_partition("dev", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("train", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("test", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)

        
def flatten(l):
    return [item for sublist in l for item in sublist]

# Usage: python minimize.py ./data_dir/ ./data_dir/ontonotes_speaker_encoding/ false
if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    do_lower_case = (sys.argv[3].lower() == 'true')
    
    print("do_lower_case", do_lower_case)
    labels = collections.defaultdict(set)
    stats = collections.defaultdict(int)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for seg_len in [384, 512]:
        minimize_language("english", labels, stats, seg_len, input_dir, output_dir, do_lower_case)
    for k, v in labels.items():
        print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
    for k, v in stats.items():
        print("{} = {}".format(k, v))
       
