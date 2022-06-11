using Microsoft.EntityFrameworkCore.Migrations;

namespace DeriHastaliklari.Migrations
{
    public partial class Tablolararasiiliski : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AddPhoto",
                columns: table => new
                {
                    ImageId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Itch = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Pain = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Cont = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Additional = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    PatientId = table.Column<int>(type: "int", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AddPhoto", x => x.ImageId);
                    table.ForeignKey(
                        name: "FK_AddPhoto_Patients_PatientId",
                        column: x => x.PatientId,
                        principalTable: "Patients",
                        principalColumn: "PatientId",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AddPhoto_PatientId",
                table: "AddPhoto",
                column: "PatientId",
                unique: true);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AddPhoto");
        }
    }
}
