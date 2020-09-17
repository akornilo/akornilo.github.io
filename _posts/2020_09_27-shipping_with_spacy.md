---
published: true
layout: post
title: "Shipping with SpaCy"
---

# Shipping with SpaCy

When it comes to Natural Language Processing, I am a huge [SpaCy](https://spacy.io/) fangirl. The library is easy to use, and the pretrained models produce great results. Beyond the existing components like Dependency Parsing and Named Entity Recognition, the library makes it easy to add custom processing steps: from [tagging additional entities using regular expressions](https://spacy.io/usage/rule-based-matching#regex) to [adding custom sentence boundary rules](https://spacy.io/usage/linguistic-features#sbd-custom) to [adding prefixes for a PERSON NER](https://spacy.io/usage/rule-based-matching#models-rules-ner). 

While SpaCy provided documentation for the various use-cases, I could not an explanation for how to package these with the model. For all the examples above, the new components were added using `nlp.add_pipe(NEW_COMPONENT)`, which modifies the existing model object with a new layer.While this set-up makes sense for experimentation, for production projects, I wanted the additional layers packaged in with the model, so I could just call `spacy.load` and have my new components available.

There is not an official approach for doing this, so in this tutorial I will outline the strategy that I used to "package" new components into a model. For this tutorial, I will focus on adding new layers to an existing model (I will touch on working with custom models at the end of this post).

![](spacy_pipeline_v2.svg)


First, we need to get a copy of an existing model SpaCy publishes the official models on github as [releases](https://github.com/explosion/spacy-models/releases/). I will take the [medium English model](https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.3.1/en_core_web_md-2.3.1.tar.gz) and untar it into a new folder (`cool_new_model`). The directory will look like:

```
en_core_web_md-2.3.1
    ├── meta.json 
    ├── setup.py
    ├── en_core_web_md/
        ├── __init__.py
        ├── en_core_web_md-2.3.1/ <-- "actual model"
            ├── ner
            ├── parser
            ├── tokenizer
            ├── ...
 
```

The files in the inner "actual model" folder contain the all the pre-trained components, I will not be touching this. The `meta.json` file is central and defines everything about the model. Most of the fields exist primarily for documentation purposes (`author`, `sources`, `description`). For this tutorial, we are interested in three of the fields: `name`, `pipelines` and `factories`.

The `name` is the name of the new model, this is the name that you would call when loading it (you can also change the `version`). We will change it to `web_custom_md`, the we will rename the directory structure to match.

```
en_web_custom_md-2.3.1
    ├── meta.json 
    ├── setup.py
    ├── en_web_custom_md/
        ├── __init__.py
        ├── en_web_custom_md-2.3.1/ <-- "actual model"
            ├── ner
            ├── parser
            ├── tokenizer
            ├── a couple of other things..
 
```

Next, we will add some new components. For simplicity, we will add the example from the SpaCy documentation that adds "titles" to an NER (e.g Ms. Kornilova). We will add the code to the `__init__.py` file that will now look like:

```
from __future__ import unicode_literals
from pathlib import Path

from spacy.util import load_model_from_init_py, get_model_meta
from spacy.language import Language
from spacy.tokens import Span


__version__ = get_model_meta(Path(__file__).parent)['version']


def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)


def expand_person_entities(doc):
    new_ents = []
    for ent in doc.ents:
        # Only check for title if it's a person and not the first token
        if ent.label_ == "PERSON" and ent.start != 0:
            prev_token = doc[ent.start - 1]
            if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
                new_ent = Span(doc, ent.start - 1, ent.end, label=ent.label)
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc
```

This will add the new function to the model package. However, we still need to associate the function with a new pipeline layer. We do this by changing the `load` method:

```

def load(**overrides):
    Language.factories['expand_person_entities'] = lambda nlp, **cfg: expand_person_entities

    return load_model_from_init_py(__file__, **overrides)
```

The `Language.factories` contain definitions for how to create the pipeline layer when a model is instantiated. Now that we know "how" to create the layer, we need to add it to the model. We modify the `pipeline` value in `meta.json` and add a new `factories` key:

```
  "pipeline":[
    "sentencizer",
    "tagger",
    "parser",
    "ner",
    "expand_person_entities"
  ],

  "factories":{
    "tagger":"tagger",
    "parser":"parser",
    "ner":"ner",
    "expand_person_entities": "expand_person_entities"
  },

```

Finally, we build the new model package by calling `pip install -e .` in the top level folder. We can now load the new model by calling `spacy.load('en_web_custom_md')`. 

```
Add the before and after example
```

You can package the model into a single file by running `python setup.py sdist`. This will produce the file `dist/en_web_cusom_md-2.3.1.tar.gz` which can be installed directly with pip.

The `__init__.py` file is also a good place to define [custom extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes). My example used a 

