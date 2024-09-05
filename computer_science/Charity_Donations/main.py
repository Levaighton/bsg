def sdn():
    # Input the names of three charities
    charity_names = []
    for i in range(3):
        charity_name = input(f"Enter the name of charity {i + 1}: ")
        charity_names.append(charity_name)

    # Display the charity names with numbers beside each name
    print("\nCharities available for donation:")
    for i, charity in enumerate(charity_names):
        print(f"{i + 1}. {charity}")

    # Set up totals for each charity donation
    charity_totals = [0, 0, 0]

    return charity_names, charity_totals


def process_donation(charity_names, charity_totals):
    while True:
        try:
            charity_choice = int(input("Choose a charity by entering the number (1, 2, or 3): "))
            if charity_choice not in [1, 2, 3]:
                raise ValueError("Invalid charity number.")
        except ValueError as e:
            print(e)
            continue
        try:
            shopping_bill = float(input("Enter the value of the customer's shopping bill: $"))
            if shopping_bill < 0:
                raise ValueError("Shopping bill must be a positive number.")
        except ValueError as e:
            print(e)
            continue
        donation = shopping_bill * 0.01
        charity_totals[charity_choice - 1] += donation
        print(f"${donation:.2f} has been donated to {charity_names[charity_choice - 1]}.")
        another = input("Do you want to process another donation? (yes/no): ").strip().lower()
        if another != "yes":
            break
    return charity_totals
def display_totals(charity_names, charity_totals):
    print("\nDonation Totals:")
    for i, total in enumerate(charity_totals):
        print(f"{charity_names[i]}: ${total:.2f}")

charity_names, charity_totals = sdn()
charity_totals = process_donation(charity_names, charity_totals)
display_totals(charity_names, charity_totals)
