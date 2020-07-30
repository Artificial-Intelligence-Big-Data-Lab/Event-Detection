

import spacy

import operator



# http://esl.fis.edu/grammar/rules/future.htm
def determine_tense_input(tagged, posextractedn):
    # text = word_tokenize(sentence)
    # tagged = pos_tag(text)

    # change is/are going to...to future
    for ww in tagged:
        if ww.pos_ == "VERB" and ww.dep_ == "aux" and (
                ww.tag_ == "VBP" or ww.tag_ == "VBZ") and ww.head.lower_ == "going":
            lll = [x for x in ww.head.rights]
            for xxx in lll:
                if xxx.pos_ == "VERB" and xxx.tag_ == "VB":
                    ww.tag_ = "MD"

    tense = {}
    future_words = [word for word in tagged if word.tag_ == "MD"]
    present_words = [word for word in tagged if word.tag_ in ["VBP", "VBZ", "VBG"]]
    pass_words = [word for word in tagged if word.tag_ in ["VBD", "VBN"]]
    inf_words = [word for word in tagged if word.tag_ in ["VB"]]  # VB	VERB	VerbForm=inf	verb, base form

    numfuture = len(future_words)
    numpresent = len(present_words)
    numpass = len(pass_words)
    numinf = len(inf_words)

    valfuture = 0
    for word in future_words:
        valfuture = valfuture + 1 / abs(posextractedn - word.i)

    valpresent = 0
    for word in present_words:
        valpresent = valpresent + 1 / abs(posextractedn - word.i)

    valpass = 0
    for word in pass_words:
        valpass = valpass + 1 / abs(posextractedn - word.i)

    valinf = 0
    for word in inf_words:
        valinf = valinf + 1 / abs(posextractedn - word.i)

    tense["future"] = valfuture
    # if valfuture > 0:
    #     tense["future"] = valfuture
    # else:
    #     if numinf > 0 and numpresent >= numpass:
    #         tense["future"] = numinf
    #     else:
    #         tense["future"] = 0

    tense["present"] = valpresent
    tense["past"] = valpass

    return (tense)


# BES	VERB		auxiliary "be"
# HVS	VERB		forms of "have"
# MD	VERB	VerbType=mod	verb, modal auxiliary
# VB	VERB	VerbForm=inf	verb, base form
# VBG	VERB	VerbForm=part Tense=pres Aspect=prog	verb, gerund or present participle
# VBP	VERB	VerbForm=fin Tense=pres	verb, non-3rd person singular present
# VBZ	VERB	VerbForm=fin Tense=pres Number=sing Person=3	verb, 3rd person singular present
# VBN	VERB	VerbForm=part Tense=past Aspect=perf	verb, past participle
# VBD	VERB	VerbForm=fin Tense=past	verb, past tense

"""
Filters the senteces of a document text, retaining only those that employ at least one future verb.
You must provide an nlp model loaded with spacy (see example in the main of this script).
Returns the filtered document as a single string.
"""
def filter_senteces(document_text, nlp_model):
    
    document=nlp_model(document_text)
    retained_sentences = []
    for sent in document.sents:
    
#        print("\n\nSENTENCE : \n" + sent.text + "\n")
        future_found = False
        for ttt in sent.noun_chunks:
    
            if ttt.root.pos_=="NOUN" and ttt.root.dep_=="nsubj":
    
#                print("  NOUN CHUNK = " + ttt.lower_)
    
                minw = sent.start
                maxw = sent.end - 1
    
                spansentence = document[minw:(maxw + 1)]
    
                tensedict = determine_tense_input(spansentence, ttt.root.i)
    
                tense = "NaN"
                tupletense = max(tensedict.items(), key=operator.itemgetter(1))  # [0]
                if tupletense[1] > 0:
                    tense = tupletense[0]
    
#                print("    Probable tense = "+ str(tense) +"\n")
                if str(tense) == 'future':
                    future_found = True
                    break
                
        if future_found:
            retained_sentences.append(sent.text)
    
    filtered_document = ' '.join(retained_sentences)
    return filtered_document


if __name__ == "__main__":
    
    
    print("Loading Spacy language models...")

    spacy_model_name_EN = 'en_core_web_lg'
    print("... " + spacy_model_name_EN + " model")
            # 'en_core_web_lg'  # 'en_' 'en_core_web_md'  'en_core_web_sm'
            ## See the spaCy page for instructions on downloading the language model.
            # English multi-task CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. Assigns word vectors, context-specific token vectors, POS tags, dependency parse and named entities.
    nlp_EN = spacy.load(spacy_model_name_EN)
    
    
    doc = "My mother has seen a dog. My mother will feed this dog after taking him home. The dog will probably like it."
#    doc = "The manufacturing sector, despite turning in its best performance for a year, is not still growing in the UK. Stock markets are rallying on improved global factory activity in the UK while whole country's economy will run fast."
    
    new_doc = filter_senteces(doc, nlp_model=nlp_EN)
    
    print('New doc:')
    print(new_doc)
    
    

#DEBUG=True
#
#
#
#print("Loading Spacy language models...")
#
#spacy_model_name_EN = 'en_core_web_lg'
#print("... " + spacy_model_name_EN + " model")
#        # 'en_core_web_lg'  # 'en_' 'en_core_web_md'  'en_core_web_sm'
#        ## See the spaCy page for instructions on downloading the language model.
#        # English multi-task CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. Assigns word vectors, context-specific token vectors, POS tags, dependency parse and named entities.
#nlp_EN = spacy.load(spacy_model_name_EN)
#
#
#
#doc=nlp_EN("The manufacturing sector, despite turning in its best performance for a year, is not still growing in the UK. Stock markets are rallying on improved global factory activity in the UK while whole country's economy will run fast.")
#
#for sent in doc.sents:
#
#    if DEBUG:
#        print("\n\nSENTENCE : \n" + sent.text + "\n")
#
#    for ttt in sent.noun_chunks:
#
#        if ttt.root.pos_=="NOUN" and ttt.root.dep_=="nsubj":
#
#            if DEBUG:
#                print("  NOUN CHUNK = " + ttt.lower_)
#
#            minw = sent.start
#            maxw = sent.end - 1
#
#            spansentence = doc[minw:(maxw + 1)]
#
#            tensedict = determine_tense_input(spansentence, ttt.root.i)
#
#            tense = "NaN"
#            tupletense = max(tensedict.items(), key=operator.itemgetter(1))  # [0]
#            if tupletense[1] > 0:
#                tense = tupletense[0]
#
#            if DEBUG == True:
#                print("    Probable tense = "+ str(tense) +"\n")
#
#
