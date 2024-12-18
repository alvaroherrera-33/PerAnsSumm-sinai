import json

# Datos JSON de entrada
data = [
    {
        "uri": "3392171",
        "question": "do braces hurt????",
        "context": "pain?\nhard to talk?",
        "answers": [
            "yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.",
            "They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.",
            "yes yes",
            "they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up.",
            "Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!!!",
            "at the begining you feel really weird and it hurts a little bit but then you get used to themis not hard to talk at all you can talk normally",
            "I had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.",
            "I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.",
            "Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain."
        ],
        "labelled_answer_spans": {
            "EXPERIENCE": [
                {"txt": ": yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain", "label_spans": [79, 377]},
                {"txt": "They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.", "label_spans": [389, 593]},
                {"txt": "they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up", "label_spans": [622, 836]},
                {"txt": "Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!", "label_spans": [848, 1080]},
                {"txt": "t the begining you feel really weird and it hurts a little bit but then you get used to themis not hard to talk at all you can talk normally", "label_spans": [1094, 1234]},
                {"txt": "had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.", "label_spans": [1247, 1418]},
                {"txt": "I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.", "label_spans": [1429, 1953]},
                {"txt": "Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain.", "label_spans": [1964, 2254]}
            ]
        },
        "raw_text": "uri: 3392171\nquestion: do braces hurt????\ncontext: pain?\nhard to talk?\nanswer_0: yes yes yes. But not horribly painful. you get used to them and they become easier to talk with. the pain is only when they get tightened and until you get used to them. They give you wax to put around the spots that hurt you when your tongue rubs against the brackets and you won't feel the pain.\nanswer_1: They hurt for a bit when you first get them.They feel tight.Then, they settle down. However, they hurt each time you get them adjusted for a few days afterward. But,dont worry,you'll get used to the pain.\nanswer_2: yes yes\nanswer_3: they hurt when u first get them but then u get used 2 them really  easily. sometimes u like drool or spit or something but unless u have a pallete expander u can talk normal. if u have 1 ur \"K's\" get all screwed up.\nanswer_4: Yes THEy HuRt WhEn YoU fIrSt GeT them but just for a few days...If you eat something hard and it gets stuck in them it mite hurt...and also after youy get them tightened..but they do not make it hard to talk, only retainers do that!!!\nanswer_5: at the begining you feel really weird and it hurts a little bit but then you get used to themis not hard to talk at all you can talk normally\nanswer_6: I had them...they hurt for a bit when you first get them...feel tight.  then, they settle down.  However, they hurt each time you get them adjusted for a few days afterward.\nanswer_7: I just got my braces a few days ago. They felt like they were loose for the first two days, and it was really hard to eat. I got over speaking difficulties quickly, although the inside of your mouth gets kind of scratched and torn- don't fret, it heals quickly enough. After that you get calouses, so it doesn't hurt anymore. I think the pain felt depends on how old you are when you get them, what kind of treatment you are getting, and how severe your bite is. Lately, I've forgotten I'd had them on. You get used to them.\nanswer_8: Intially they are painful...also depending on your age and the condition your teeth are in. I have had my for almost 2 1/2 years and my first year was painful. Now I forget I have them...they are coming off soon and I can't wait! I recommend Aleve, its really helpful in relieving the pain.\n"
    }
]

def verify_spans(data):
    for entry in data:
        raw_text = entry.get("raw_text", "")
        labelled_spans = entry.get("labelled_answer_spans", {})

        print(f"\nVerifying spans for URI: {entry['uri']}")

        for label, spans in labelled_spans.items():
            for span in spans:
                span_start, span_end = span["label_spans"]
                extracted_text = raw_text[span_start:span_end]

                print(span_start, span_end)
                if extracted_text != span["txt"]:
                    print(f"  Mismatch in label '{label}':\n    Expected: '{span['txt']}'\n    Found:    '{extracted_text}'")
                else:
                    print(f"  Span for label '{label}' is correct.")

# Llamar a la función
verify_spans(data)
