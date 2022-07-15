def generate_FEN(prediction_labels):
    FEN_string = ""
    empty_counter = 0
    column_counter = 0
    i = 0
    for label in prediction_labels:
        print(column_counter)
        if column_counter == 8:
            FEN_string = FEN_string + "/"
            column_counter = 0
        if label == "empty":
            empty_counter = empty_counter + 1
            if i + 1 < 64:
                if prediction_labels[i+1] != "empty":
                    FEN_string = FEN_string + str(empty_counter)
                    empty_counter = 0
                elif column_counter == 7:
                    FEN_string = FEN_string + str(empty_counter)
                    empty_counter = 0
                    column_counter = column_counter + 1
            else:
                FEN_string = FEN_string + str(empty_counter)
        elif label == "bb":
            FEN_string = FEN_string + "b"
            empty_counter = 0
        elif label == "bk":
            FEN_string = FEN_string + "k"
            empty_counter = 0
        elif label == "bn":
            FEN_string = FEN_string + "n"
            empty_counter = 0
        elif label == "bp":
            FEN_string = FEN_string + "p"
            empty_counter = 0
        elif label == "bq":
            FEN_string = FEN_string + "q"
            empty_counter = 0
        elif label == "br":
            FEN_string = FEN_string + "r"
            empty_counter = 0
        elif label == "wb":
            FEN_string = FEN_string + "B"
            empty_counter = 0
        elif label == "wk":
            FEN_string = FEN_string + "K"
            empty_counter = 0
        elif label == "wn":
            FEN_string = FEN_string + "N"
            empty_counter = 0
        elif label == "wp":
            FEN_string = FEN_string + "P"
            empty_counter = 0
        elif label == "wq":
            FEN_string = FEN_string + "Q"
            empty_counter = 0
        elif label == "wr":
            FEN_string = FEN_string + "R"
            empty_counter = 0
        i = i + 1
        if column_counter != 8:
            column_counter = column_counter + 1

    return FEN_string