from preprocess import preprocess_movielens

def main():
    user_seq = preprocess_movielens()
    print(user_seq)

if __name__ == "__main__":
    main()