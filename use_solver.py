from solver import create_nng, nng_base

SIZE = 5

row_res, col_res = create_nng.create_nng_from_image(
    path = 'ressources/Cats.jpg',
    size=SIZE,
    show_image=True
)

print(f'row_res = {row_res}')
print(f'col_res = {col_res}')

nng = nng_base.NNGStupidRec(
    size=SIZE,
    row_restrictions=row_res,
    col_restrictions=col_res,
    save_history=True)

nng.solve()