""" Data processors and helpers """
import csv
import json
import logging
import os

import html
import sys

from tqdm import tqdm
from transformers import glue_processors, glue_output_modes
from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from sklearn.model_selection import train_test_split

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        # processor = glue_processors[task]()
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,
                                       truncation = True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
            X_test = []
            y_test = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                X_test.append(line[0])
                y_test.append(line[1])

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")
        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line[0]) is not str:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
            else:
                if i == 0:
                    continue
                text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        X = []
        Y = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            X.append(line[0])
            Y.append(line[1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

glue_tasks = ["sst-2"]

class ImdbProcessor(DataProcessor):
    """Processor for the PubMed data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "train_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "train_42.json")) as json_file:
            [X_train, y_train] = json.load(json_file)
        # X, Y = [], []
        # lines = self._read_csv(os.path.join(data_dir, "train", "train.tsv"))
        # for line in lines[1:]:
        #     # X.append(",".join(line[1:]).rstrip())
        #     X.append(",".join(line[:-1]).rstrip())
        #     # Y.append(line[0])
        #     Y.append(line[-1])
        train_examples = self._create_examples(zip(X_train, y_train), "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
            self._train_dev_split(data_dir)
        with open(os.path.join(data_dir, "dev_42.json")) as json_file:
            [X_val, y_val] = json.load(json_file)
        dev_examples = self._create_examples(zip(X_val, y_val), "dev")
        return dev_examples

    # def get_contrast_examples(self, file=None, ori=False, data_dir=IMDB_CONTR_DATA_DIR):
    #     # if not os.path.isfile(os.path.join(data_dir, "dev_42.json")):
    #     #     self._train_dev_split(data_dir)
    #     # with open(os.path.join(data_dir, "dev_42.json")) as json_file:
    #     #     [X_val, y_val] = json.load(json_file)
    #     prefix='original' if ori else 'contrast'
    #     X, Y = [], []
    #     lines = self._read_csv(os.path.join(data_dir, "{}_{}.tsv".format(file, prefix)))
    #     labelname2int={"Positive":"1", "Negative":"0"}
    #     for i, line in enumerate(lines):
    #         if i == 0:
    #             continue
    #         # X.append(",".join(line[1:]).rstrip())
    #         X.append(",".join(line).rstrip().split('\t')[1])
    #         # Y.append(line[0])
    #         Y.append(labelname2int[",".join(line).rstrip().split('\t')[0]])
    #     dev_examples = self._create_examples(zip(X, Y), "{}_{}".format(file, prefix))
    #     return dev_examples

    def get_test_examples(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, "test.json")):
            with open(os.path.join(data_dir, "test.json")) as json_file:
                [X_test, y_test] = json.load(json_file)
        else:

            X_test, y_test = [], []
            lines = self._read_csv(os.path.join(data_dir, "test", "test.tsv"))
            for line in lines[1:]:
                # X.append(",".join(line[1:]).rstrip())
                X_test.append(",".join(line[:-1]).rstrip())
                # Y.append(line[0])
                y_test.append(line[-1])

            # Write the dev set into a json file (for this seed)
            with open(os.path.join(data_dir, "test.json"), "w") as f:
                json.dump([X_test, y_test], f)

        test_examples = self._create_examples(zip(X_test, y_test), "test")

        return test_examples

    def get_augm_examples(self, X, y):
        return self._create_examples(zip(X,y), "augm")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "augm":
                if type(line) is not str:
                    text_a = line[0][0]
                else:
                    text_a = line[0]
                label = line[1]
            else:
                text_a, label = line
                # if dom != -1:
                #     label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _train_dev_split(self, data_dir, seed=42):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self._read_csv(os.path.join(data_dir, "train", "train.tsv"))
        for line in lines[1:]:
            # X.append(",".join(line[1:]).rstrip())
            X.append(",".join(line[:-1]).rstrip())
            # Y.append(line[0])
            Y.append(line[-1])

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=seed)

        # Write the train set into a json file (for this seed)
        with open(os.path.join(data_dir, "train_{}.json".format(seed)), "w") as f:
            json.dump([X_train , Y_train], f)

        # Write the dev set into a json file (for this seed)
        with open(os.path.join(data_dir, "dev_{}.json".format(seed)), "w") as f:
            json.dump([X_val , Y_val], f)

        return

processors = {
    "sst-2": Sst2Processor,
    "imdb": ImdbProcessor,
}

output_modes = {
    "sst-2": "classification",
    "imdb": "classification",
}
